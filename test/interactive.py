import os
import sys
import math
import argparse
import typing

import numpy as np
import imageio
from PIL import Image
import cv2
import torch
from torchvision import utils as vutils

from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QPainter
from PyQt5.QtGui import QPen
from PyQt5.QtGui import QBrush
from PyQt5.QtCore import Qt

from srwarp import transform
from srwarp import warp

def np2tensor(x: np.array) -> torch.Tensor:
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x)
    with torch.no_grad():
        while x.dim() < 4:
            x.unsqueeze_(0)

        x = x.float() / 255

    return x

def tensor2np(x: torch.Tensor) -> np.array:
    with torch.no_grad():
        x = 255 * x
        x = x.round().clamp(min=0, max=255).byte()
        x = x.squeeze(0)

    x = x.cpu().numpy()
    x = np.transpose(x, (1, 2, 0))
    x = np.ascontiguousarray(x)
    return x


class Interactive(QMainWindow):

    def __init__(
            self,
            app: QApplication,
            img: str,
            kernel_size: int,
            single: bool,
            load_m: bool,
            pretrained: str) -> None:

        super().__init__()
        self.setStyleSheet('background-color: gray;')
        self.margin = 300
        img = Image.open(img)
        self.img = np.array(img)
        self.img_tensor = np2tensor(self.img).cuda()
        self.img_h = self.img.shape[0]
        self.img_w = self.img.shape[1]
        if single:
            self.offset_h = 5
            self.offset_w = 5
        else:
            self.offset_h = self.margin
            self.offset_w = self.img_w + 2 * self.margin

        window_h = self.img_h + 2 * self.margin
        window_w = 2 * self.img_w + 3 * self.margin

        monitor_resolution = app.desktop().screenGeometry()
        screen_h = monitor_resolution.height()
        screen_w = monitor_resolution.width()

        screen_offset_h = (screen_h - window_h) // 2
        screen_offset_w = (screen_w - window_w) // 2

        self.setGeometry(screen_offset_w, screen_offset_h, window_w, window_h)
        self.reset_cps()
        if load_m:
            self.cps = torch.load('interactive_cps.pth')

        self.line_order = ('tl', 'tr', 'br', 'bl')
        self.grab = None
        self.shift = False

        #self.inter = cv2.INTER_CUBIC
        self.inter = cv2.INTER_LINEAR
        self.backend = 'core'
        self.multi_scale = False

        self.single = single
        self.backup = None
        self.backup_img = None
        self.backup_dump = None

        # For debugging
        self.update()

        torch.set_printoptions(precision=12, linewidth=160, sci_mode=False)
        self.mouse_pos = None
        self.status = self.statusBar()
        return

    def reset_cps(self) -> None:
        self.cps = {
            'tl': (0, 0),
            'tr': (0, self.img_w - 1),
            'bl': (self.img_h - 1, 0),
            'br': (self.img_h - 1, self.img_w - 1),
        }
        return

    def keyReleaseEvent(self, e) -> None:
        if e.key() == Qt.Key_Shift:
            self.shift = False

        return

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Escape:
            self.close()

        if e.key() == Qt.Key_Shift:
            self.shift = True

        if e.key() == Qt.Key_I:
            if self.inter == cv2.INTER_CUBIC:
                self.inter = cv2.INTER_NEAREST
            elif self.inter == cv2.INTER_NEAREST:
                self.inter = cv2.INTER_LINEAR
            else:
                self.inter = cv2.INTER_CUBIC

            self.update()
        elif e.key() == Qt.Key_M:
            if self.backend == 'opencv':
                self.backend = 'core'
            elif self.backend == 'core':
                self.backend = 'opencv'

            self.update()
        elif e.key() == Qt.Key_S:
            self.update()
        elif e.key() == Qt.Key_R:
            self.reset_cps()
            self.update()
        elif e.key() == Qt.Key_D:
            if self.backup_dump is not None:
                print('Saving debugging logs...')
                torch.save(self.backup_dump, 'example/dump.pth')
                print('Saved')
        elif e.key() == Qt.Key_P:
            m, _, _ = self.get_matrix()
            torch.save(self.cps, 'interactive_cps.pth')
            print(m)
        elif e.key() == Qt.Key_V:
            if self.backup_img is not None:
                imageio.imwrite('example/warped.png', self.backup_img)

        return

    def mousePressEvent(self, e) -> None:
        is_left = e.buttons() & Qt.LeftButton
        if is_left:
            threshold = 20
            min_dist = 987654321
            for key, val in self.cps.items():
                y, x = val
                dy = e.y() - y - self.offset_h
                dx = e.x() - x - self.offset_w
                dist = dy ** 2 + dx ** 2
                if dist < min_dist:
                    min_dist = dist
                    self.grab = key

            if min_dist > threshold ** 2:
                self.grab = None

        return

    def get_matrix(self) -> typing.Tuple[np.array, int, int]:
        points_from = np.array([
            [0, 0],
            [self.img_w - 1, 0],
            [0, self.img_h - 1],
            [self.img_w - 1, self.img_h - 1],
        ]).astype(np.float32)
        points_to = np.array([
            [self.cps['tl'][1], self.cps['tl'][0]],
            [self.cps['tr'][1], self.cps['tr'][0]],
            [self.cps['bl'][1], self.cps['bl'][0]],
            [self.cps['br'][1], self.cps['br'][0]],
        ]).astype(np.float32)
        m = cv2.getPerspectiveTransform(points_from, points_to)
        m = torch.from_numpy(m)

        dummy = torch.zeros(1, 1, self.img_h, self.img_w)
        m, sizes, offsets = transform.compensate_matrix(dummy, m, exact=False)
        return m, sizes, offsets

    def mouseMoveEvent(self, e) -> None:
        if self.grab is not None:
            y_old, x_old = self.cps[self.grab]
            y_new = e.y() - self.offset_h
            x_new = e.x() - self.offset_w
            self.cps[self.grab] = (y_new, x_new)
            if self.shift:
                tb = self.grab[0]
                lr = self.grab[1]
                anchor = None
                for key, val in self.cps.items():
                    if not (tb in key or lr in key):
                        anchor = val
                        break

                for key, val in self.cps.items():
                    if key == self.grab:
                        continue

                    if tb in key:
                        self.cps[key] = (y_new, anchor[1])

                    if lr in key:
                        self.cps[key] = (anchor[0], x_new)

            is_convex = True
            #cross = None
            for i, pos in enumerate(self.line_order):
                y1, x1 = self.cps[pos]
                y2, x2 = self.cps[self.line_order[(i + 1) % 4]]
                y3, x3 = self.cps[self.line_order[(i + 2) % 4]]
                dx1 = x2 - x1
                dy1 = y2 - y1
                dx2 = x3 - x2
                dy2 = y3 - y2
                cross_new = dx1 * dy2 - dy1 * dx2
                if cross_new < 1500:
                    is_convex = False
                    break

            if not is_convex:
                self.cps[self.grab] = (y_old, x_old)

        self.mouse_pos = (e.y(), e.x())
        self.update()
        return

    def mouseReleaseEvent(self, e) -> None:
        if self.grab is not None:
            self.grab = None

        return

    @torch.no_grad()
    def paintEvent(self, e) -> None:
        if self.inter == cv2.INTER_NEAREST:
            inter_method = 'Nearest'
        elif self.inter == cv2.INTER_LINEAR:
            inter_method = 'Bilinear'
        elif self.inter == cv2.INTER_CUBIC:
            inter_method = 'Bicubic'

        self.setWindowTitle(
            'Interpolation: {} / backend: {} / Multi-scale: {}'.format(
                inter_method, self.backend, self.multi_scale,
            )
        )

        qp = QPainter()
        qp.begin(self)
        if not self.single:
            qimg = QImage(
                self.img,
                self.img_w,
                self.img_h,
                3 * self.img_w,
                QImage.Format_RGB888,
            )
            qpix = QPixmap(qimg)
            qp.drawPixmap(self.margin, self.margin, self.img_w, self.img_h, qpix)

        m, sizes, offsets = self.get_matrix()
        h_new, w_new = sizes
        dump = {}
        if self.backend == 'opencv':
            m = m.numpy()
            y = cv2.warpPerspective(
                self.img,
                m,
                (w_new, h_new),
                flags=self.inter,
                #borderMode=cv2.BORDER_REFLECT,
            )
            dump = None
        elif self.backend == 'core':
            y = warp.warp_by_function(
                self.img_tensor,
                m,
                f_inverse=False,
                sizes=(h_new, w_new),
                adaptive_grid=(self.inter != cv2.INTER_CUBIC),
                regularize=(self.inter != cv2.INTER_NEAREST),
                fill=-255,
                dump=dump,
            )
            y = tensor2np(y)
            self.backup_img = y

        qimg_warp = QImage(y, w_new, h_new, 3 * w_new, QImage.Format_RGB888)
        qpix_warp = QPixmap(qimg_warp)
        qp.drawPixmap(
            self.offset_w - offsets[1],
            self.offset_h - offsets[0],
            w_new,
            h_new,
            qpix_warp,
        )
        center_y = self.offset_h + self.img_h // 2
        center_x = self.offset_w + self.img_w // 2

        pen_blue = QPen(Qt.blue, 5)
        pen_white = QPen(Qt.white, 10)
        text_size = 20
        for key, val in self.cps.items():
            y, x = val
            y = y + self.offset_h
            x = x + self.offset_w
            qp.setPen(pen_blue)
            qp.drawPoint(x, y)
            qp.setPen(pen_white)
            dy = y - center_y
            dx = x - center_x
            dl = 1 + math.sqrt(dy ** 2 + dx ** 2) / 10
            qp.drawText(
                x + (dx / dl) - text_size // 2,
                y + (dy / dl) - text_size // 2,
                text_size,
                text_size,
                int(Qt.AlignCenter),
                key,
            )

        msg = 'M: [[{:.3f} {:.3f} {:.3f}] [{:.3f} {:.3f} {:.3f}] [{:.3f} {:.3f} {:.3f}]]'.format(
            m[0, 0], m[0, 1], m[0, 2],
            m[1, 0], m[1, 1], m[1, 2],
            m[2, 0], m[2, 1], m[2, 2],
        )
        if self.mouse_pos is not None and dump is not None:
            mouse_y, mouse_x = self.mouse_pos
            py = mouse_y - self.offset_h + offsets[0]
            px = mouse_x - self.offset_w + offsets[1]
            msg += '\tx: {} / y: {}'.format(px, py)
            if px >= 0 and py >= 0 and px < sizes[1] and py < sizes[0]:
                n = sizes[0] * sizes[1]
                idx = py * sizes[1] + px
                grid_raw = dump['grid_raw']
                yi = dump['yi']
                if (yi == idx).sum() > 0:
                    dummy_grid_raw = grid_raw.new_zeros(2, n)
                    dummy_grid_raw[..., yi] = grid_raw
                    dummy_grid_raw_x = dummy_grid_raw[0, idx].item()
                    dummy_grid_raw_y = dummy_grid_raw[1, idx].item()
                    msg += '\tsx: {:.3f} / sy: {:.3f}'.format(
                        dummy_grid_raw_x, dummy_grid_raw_y,
                    )
                    if 'j' in dump and dump['j'] is not None:
                        j = dump['j']
                        dummy_du = j[0].new_zeros(2, n)
                        dummy_dv = j[1].new_zeros(2, n)
                        dummy_du[..., yi] = j[0]
                        dummy_dv[..., yi] = j[1]
                        
                        a = dummy_du[0][idx].item()
                        b = dummy_dv[0][idx].item()
                        c = dummy_du[1][idx].item()
                        d = dummy_dv[1][idx].item()
                        msg += '\tJ: [[{:.3f} {:.3f}] [{:.3f} {:.3f}]]'.format(
                            a, b, c, d,
                        )

                        det = a * d - b * c
                        if abs(det) > 1e-12:
                            msg += ' J_inv: [[{:.3f} {:.3f}] [{:.3f} {:.3f}]]'.format(
                                d / det, -b / det, -c / det, a / det,
                            )
                            msg += ' Det.: {:.3f}'.format(1 / abs(det))

                        a_inv = dump['a']
                        b_inv = dump['b']
                        omega = dump['omega']
                        dummy_a_inv = a_inv.new_zeros(n)
                        dummy_b_inv = b_inv.new_zeros(n)
                        dummy_omega = omega.new_zeros(n)
                        dummy_a_inv[yi] = a_inv
                        dummy_b_inv[yi] = b_inv
                        dummy_omega[yi] = omega

                        ea = (1 / dummy_a_inv[idx]).item()
                        eb = (1 / dummy_b_inv[idx]).item()
                        eo = dummy_omega[idx].item()
                        msg += '\tA: {:.3f} B: {:.3f} omega: {:.3f}'.format(
                            ea, eb, eo,
                        )
                        cx = self.margin + round(dummy_grid_raw_x)
                        cy = self.margin + round(dummy_grid_raw_y)
                        self.draw_grid(qp, cx, cy, ea, eb, eo)

                        dump_save = {
                            'ux': a,
                            'vx': b,
                            'uy': c,
                            'vy': d,
                            'a': ea,
                            'b': eb,
                            'omega': eo,
                        }
                        torch.save(dump_save, 'dump.pth')

        self.status.showMessage(msg)
        qp.end()
        return

    def draw_grid(
            self,
            painter: QPainter,
            cx: int,
            cy: int,
            a: float,
            b: float,
            omega: float) -> None:

        ex = (a * math.cos(omega), a * math.sin(omega))
        ey = (-b * math.sin(omega), b * math.cos(omega))

        l = 100
        xx_right = int(cx + l * ex[0])
        xy_right = int(cy + l * ex[1])
        yx_right = int(cx + l * ey[0])
        yy_right = int(cy + l * ey[1])

        pen_red = QPen(Qt.red, 3)
        painter.setPen(pen_red)
        painter.drawLine(cx, cy, xx_right, xy_right)

        pen_blue = QPen(Qt.blue, 3)
        painter.setPen(pen_blue)
        painter.drawLine(cx, cy, yx_right, yy_right)
        
        ex = (1, 0)
        ey = (0, 1)

        xx_right = int(cx + l * ex[0])
        xy_right = int(cy + l * ex[1])
        yx_right = int(cx + l * ey[0])
        yy_right = int(cy + l * ey[1])

        pen_red = QPen(Qt.red, 3, Qt.DotLine)
        painter.setPen(pen_red)
        painter.drawLine(cx, cy, xx_right, xy_right)

        pen_blue = QPen(Qt.blue, 3, Qt.DotLine)
        painter.setPen(pen_blue)
        painter.drawLine(cx, cy, yx_right, yy_right)

        return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, default='../../../dataset/benchmark/set5/HR/butterfly.png')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--single', action='store_true')
    parser.add_argument('--load_m', action='store_true')
    parser.add_argument('--kernel_size', type=int, default=4)
    parser.add_argument('--pretrained', type=str)
    cfg = parser.parse_args()
    np.set_printoptions(precision=5, linewidth=200, suppress=True)
    app = QApplication(sys.argv)
    sess = Interactive(
        app,
        cfg.img,
        cfg.kernel_size,
        cfg.single,
        cfg.load_m,
        cfg.pretrained,
    )

    if cfg.full:
        sess.showFullScreen()
    else:
        sess.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
