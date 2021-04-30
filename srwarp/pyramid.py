'''
Build an image pyramid.
'''

import sys
import math
import random
import typing

import numpy as np
import cv2
import torch

from srwarp import transform

def deg2rad(theta: float) -> float:
    ret = math.pi * (theta / 180)
    return ret

def get_normal(theta: float, phi: float) -> torch.Tensor:
    theta = deg2rad(theta)
    phi = deg2rad(phi)
    ret = torch.Tensor([
        math.sin(phi) * math.cos(theta),
        math.sin(phi) * math.sin(theta),
        math.cos(phi),
    ])
    return ret


class Pyramid(object):

    def __init__(self, height: float, width: float) -> None:
        self.__height = height
        self.__width = width
        self.__directions = torch.Tensor([
            [height // 2, width // 2, 1],
            [-height // 2, width // 2, 1],
            [-height // 2, -width // 2, 1],
            [height // 2, -width // 2, 1],
        ])
        return

    @torch.no_grad()
    def solve_t(
            self,
            normal: torch.Tensor,
            center: torch.Tensor) -> typing.List[float]:

        num = normal.dot(center)
        den = self.__directions.matmul(normal)
        t = num / den
        return t

    @torch.no_grad()
    def solve_p(
            self,
            normal: torch.Tensor,
            center: torch.Tensor,
            transpose: bool=False) -> torch.Tensor:

        num = normal.dot(center)
        scales = torch.Tensor([num / normal.dot(d) for d in self.__directions])
        scales = scales.view(-1, 1)
        points = scales * self.__directions
        points.t_()
        points = torch.cat((points, torch.ones(1, points.size(1))), dim=0)

        dz = num / normal[2]
        cos = normal[2]
        sin = torch.sqrt(1 - cos.pow(2))
        u = torch.Tensor([normal[1], -normal[0]])
        u_norm = u.norm()
        if u_norm > 1e-6:
            u /= u_norm

        u0sin = u[0] * sin
        u1sin = u[1] * sin
        u0u1 = u[0] * u[1] * (1 - cos)
        m = torch.Tensor([
            [cos + (1 - cos) * u[0]**2, u0u1, u1sin, 0],
            [u0u1, cos + (1 - cos) * u[1]**2, -u0sin, 0],
            [-u1sin, u0sin, cos, -dz],
            [0, 0, 0, 1],
        ])
        points = m.matmul(points)
        points = points[:2]
        if transpose:
            points.t_()

        return points

    @torch.no_grad()
    def solve_m(
            self,
            normal: torch.Tensor,
            center: torch.Tensor) -> torch.Tensor:

        points = self.solve_p(normal, center, transpose=True)
        points_from = np.array([
            [self.__width - 1, self.__height - 1],
            [0, self.__height - 1],
            [0, 0],
            [self.__width - 1, 0],
        ])
        points_from = points_from.astype(np.float32)
        points_to = points[:, :2]
        points_to = points_to.numpy()
        m = cv2.getPerspectiveTransform(points_from, points_to)
        m = torch.from_numpy(m)
        return m

    @torch.no_grad()
    def get_random_m(
            self,
            phi_min: float=0,
            phi_max: float=0.2,
            z_min: float=0.5,
            z_max: float=4,
            random_aspect: bool=False,
            inverse: bool=False) -> torch.Tensor:

        phi = random.uniform(phi_min, phi_max)
        theta = random.uniform(0, 360)
        normal = get_normal(theta, phi)
        z = random.uniform(z_min, z_max)
        center = torch.Tensor([0, 0, z])
        m = self.solve_m(normal, center)
        if random_aspect:
            sx = random.uniform(0.8, 1.2)
            sy = random.uniform(0.8, 1.2)
            m = transform.compensate_scale(m, sx, sy=sy)

        if inverse:
            m = transform.inverse_3x3(m)

        return m
