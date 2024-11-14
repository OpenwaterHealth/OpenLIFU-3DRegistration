import pytest
import numpy as np
import cv2
import vtk
import os
from pathlib import Path
from render_and_landmarks import MeshRenderer, LandmarkDetector

TEST_DATA_DIR = Path("tests/test_data")
TEST_MESH = TEST_DATA_DIR / "test_face.stl"
TEST_RENDER = TEST_DATA_DIR / "test_render.png"
TEST_LANDMARK = TEST_DATA_DIR / "test_landmark.png"

class TestMeshRenderer:
    @pytest.fixture
    def test_mesh(self):
        """Load test mesh."""
        reader = vtk.vtkSTLReader()
        reader.SetFileName(str(TEST_MESH))
        reader.Update()
        return reader.GetOutput()

    def test_render_mesh_at_angles(self, test_mesh):
        """Test mesh rendering at specific angles."""
        renderer = MeshRenderer()
        angles = [(0, 0, 0), (90, 0, 0)]
        images = renderer.render_mesh_at_angles(test_mesh, angles)
        
        assert len(images) == len(angles)
        assert images[0].shape == (512, 512, 3)
        
        # Compare with reference render
        ref_image = cv2.imread(str(TEST_RENDER))
        assert np.allclose(images[0], ref_image, rtol=1e-3, atol=10)
        
        # Test output image properties
        for img in images:
            assert img.dtype == np.uint8
            assert img.shape == (512, 512, 3)
            assert np.max(img) > 0  # Image not empty

    def test_render_with_skin_color(self, test_mesh):
        """Test rendering with custom skin color."""
        renderer = MeshRenderer()
        skin_color = (0.8, 0.6, 0.5)
        images = renderer.render_mesh_at_angles(test_mesh, [(0, 0, 0)], skin_color=skin_color)
        
        img = images[0]
        # Check if dominant color is close to skin tone
        mean_color = np.mean(img[img.sum(axis=2) > 0], axis=0)
        expected_color = np.array(skin_color)[::-1] * 255  # Convert to BGR
        assert np.allclose(mean_color, expected_color, rtol=0.2)

class TestLandmarkDetector:
    @pytest.fixture
    def test_image(self):
        """Load test face image with known landmarks."""
        return cv2.imread(str(TEST_RENDER))

    def test_detect_face_landmarks(self, test_image):
        """Test facial landmark detection on rendered image."""
        detector = LandmarkDetector()
        index, annotated_image, landmarks_2d = detector.detect_face_landmarks([test_image])
        
        assert index == 0
        assert annotated_image is not None
        assert len(landmarks_2d) > 0
        
        # Compare with reference landmarks
        ref_landmarks = np.load(str(TEST_DATA_DIR / "reference_landmarks.npy"))
        assert np.allclose(np.array(landmarks_2d), ref_landmarks, rtol=1e-2)
        
        # Save annotated image for visual verification
        cv2.imwrite(str(TEST_DATA_DIR / "output_landmarks.png"), annotated_image)

    def test_landmark_detection_multiple_angles(self, test_image):
        """Test landmark detection on multiple angle renders."""
        detector = LandmarkDetector()
        # Create rotated versions
        rotated = cv2.rotate(test_image, cv2.ROTATE_90_CLOCKWISE)
        images = [test_image, rotated]
        
        index, annotated_image, landmarks_2d = detector.detect_face_landmarks(images)
        
        assert 0 <= index < len(images)
        assert len(landmarks_2d) > 0
        
        # Verify face detected in front-facing image
        assert index == 0  # Assuming front-facing has better detection

    def test_unproject_2d_to_3d(self):
        """Test 2D to 3D coordinate unprojection."""
        detector = LandmarkDetector()
        renderer = vtk.vtkRenderer()
        camera = renderer.GetActiveCamera()
        
        test_points = [(256, 256), (100, 100)]  # Center and arbitrary point
        points_3d = detector.unproject_2d_to_3d(camera, renderer, test_points)
        
        assert len(points_3d) == len(test_points)
        assert all(len(p) == 3 for p in points_3d)
        
        # Compare with known good unprojection
        ref_points = np.load(str(TEST_DATA_DIR / "reference_3d_points.npy"))
        assert np.allclose(np.array(points_3d), ref_points, rtol=1e-2)

    def test_render_landmarks_in_volume(self):
        """Test rendering 3D landmarks in volume."""
        detector = LandmarkDetector()
        renderer = vtk.vtkRenderer()
        
        test_landmarks = [(0, 0, 0), (10, 10, 10)]
        modified_renderer = detector.render_landmarks_in_volume(
            renderer, test_landmarks, landmark_color=(1, 0, 0)
        )
        
        assert modified_renderer.GetActors().GetNumberOfItems() == len(test_landmarks)
        
        # Verify landmark properties
        actors = modified_renderer.GetActors()
        actors.InitTraversal()
        for _ in range(actors.GetNumberOfItems()):
            actor = actors.GetNextActor()
            color = actor.GetProperty().GetColor()
            assert color == (1, 0, 0)

if __name__ == "__main__":
    pytest.main([__file__])