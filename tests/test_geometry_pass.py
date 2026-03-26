import unittest

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


if __name__ == "__main__":
    unittest.main()
