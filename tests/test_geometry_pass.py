import unittest
from types import SimpleNamespace

import torch

from nvdiffrast_mesh_renderer.geometry_pass import GeometryPassRenderer


class GeometryPassFacingTests(unittest.TestCase):
    def setUp(self):
        self.renderer = GeometryPassRenderer(glctx=None, config=None)
        self.positions = torch.tensor(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=torch.float32,
        )

    def test_split_uses_camera_side_of_face_plane(self):
        faces = torch.tensor([[0, 1, 2], [0, 2, 1]], dtype=torch.int32)
        face_normals = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=torch.float32,
        )
        camera_position = torch.tensor([0.0, 0.0, 2.0], dtype=torch.float32)

        front_tri, front_normals, back_tri, back_normals = self.renderer._split_triangles_by_facing(
            faces,
            self.positions,
            face_normals,
            camera_position,
        )

        self.assertTrue(torch.equal(front_tri, faces[:1]))
        self.assertTrue(torch.equal(front_normals, face_normals[:1]))
        self.assertTrue(torch.equal(back_tri, faces[1:]))
        self.assertTrue(torch.equal(back_normals, face_normals[1:]))

    def test_split_marks_face_as_back_when_camera_is_behind_plane(self):
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        face_normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        camera_position = torch.tensor([0.0, 0.0, -2.0], dtype=torch.float32)

        front_tri, front_normals, back_tri, back_normals = self.renderer._split_triangles_by_facing(
            faces,
            self.positions,
            face_normals,
            camera_position,
        )

        self.assertEqual(front_tri.numel(), 0)
        self.assertEqual(front_normals.numel(), 0)
        self.assertTrue(torch.equal(back_tri, faces))
        self.assertTrue(torch.equal(back_normals, face_normals))

    def test_split_uses_orthographic_forward_direction(self):
        faces = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        face_normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
        camera = SimpleNamespace(
            projection_type="orthographic",
            forward=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32),
            position=torch.tensor([0.0, 0.0, -2.0], dtype=torch.float32),
        )

        front_tri, front_normals, back_tri, back_normals = self.renderer._split_triangles_by_facing(
            faces,
            self.positions,
            face_normals,
            camera,
        )

        self.assertTrue(torch.equal(front_tri, faces))
        self.assertTrue(torch.equal(front_normals, face_normals))
        self.assertEqual(back_tri.numel(), 0)
        self.assertEqual(back_normals.numel(), 0)


class GeometryPassCoordinateTests(unittest.TestCase):
    def test_camera_relative_projection_is_translation_invariant(self):
        renderer = GeometryPassRenderer(glctx=None, config=None)
        local_positions = torch.tensor(
            [
                [-0.07, -0.11, -0.35],
                [0.07, -0.11, -0.35],
                [0.0, 0.12, -0.35],
            ],
            dtype=torch.float32,
        )
        camera_position = torch.tensor([998.5, -829.4, 16.2], dtype=torch.float32)
        rotation = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        projection = torch.tensor(
            [
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, -1.1, -0.2],
                [0.0, 0.0, -1.0, 0.0],
            ],
            dtype=torch.float32,
        )
        view = torch.eye(4, dtype=torch.float32)
        view[:3, :3] = rotation
        view[:3, 3] = -(rotation @ camera_position)
        camera = SimpleNamespace(
            position=camera_position,
            view=view,
            proj=projection,
        )

        clip, view_attr = renderer._camera_relative_positions(
            local_positions + camera_position,
            camera,
        )
        expected_view = local_positions @ rotation.t()
        expected_view_h = torch.cat(
            [expected_view, torch.ones((3, 1), dtype=torch.float32)],
            dim=-1,
        )
        expected_clip = expected_view_h @ projection.t()

        self.assertTrue(torch.allclose(view_attr, expected_view, atol=1e-4, rtol=0.0))
        self.assertTrue(torch.allclose(clip, expected_clip, atol=1e-4, rtol=0.0))


if __name__ == "__main__":
    unittest.main()
