import pytest
import SimpleITK as sitk
import numpy as np
import vtk
import os
from unittest.mock import MagicMock, patch

from surface_mesh_extractor import MRIProcessor, VTKConverter, SurfaceMeshExtractor

@pytest.fixture
def sample_image():
    """Create a sample 3D image for testing."""
    # Create a 32x32x32 volume with a sphere in the middle
    size = [32, 32, 32]
    image = sitk.Image(size, sitk.sitkFloat32)
    
    for z in range(size[2]):
        for y in range(size[1]):
            for x in range(size[0]):
                # Create a sphere
                if (x-16)**2 + (y-16)**2 + (z-16)**2 < 100:
                    image[x,y,z] = 1.0
                else:
                    image[x,y,z] = 0.0
    
    return image

class TestMRIProcessor:
    def test_read_mri_volume(self, tmp_path):
        """Test reading an MRI volume from a file."""
        processor = MRIProcessor()
        
        # Create a temporary test image
        test_image = sitk.Image([10, 10, 10], sitk.sitkFloat32)
        test_file = tmp_path / "test_image.nii.gz"
        sitk.WriteImage(test_image, str(test_file))
        
        # Read the image
        result = processor.read_mri_volume(str(test_file))
        assert isinstance(result, sitk.Image)
        assert result.GetSize() == (10, 10, 10)

    def test_apply_otsu_threshold_global(self, sample_image):
        """Test global Otsu thresholding."""
        processor = MRIProcessor()
        binary_image, threshold = processor.apply_otsu_threshold_global(sample_image)
        
        assert isinstance(binary_image, sitk.Image)
        assert isinstance(threshold, float)
        assert threshold > 0
        
        # Check binary nature of output
        array = sitk.GetArrayFromImage(binary_image)
        assert np.array_equal(np.unique(array), np.array([0, 1]))

    def test_apply_otsu_threshold_slice_by_slice(self, sample_image):
        """Test slice-by-slice Otsu thresholding."""
        processor = MRIProcessor()
        binary_image, thresholds = processor.apply_otsu_threshold_slice_by_slice(sample_image)
        
        assert isinstance(binary_image, sitk.Image)
        assert isinstance(thresholds, list)
        assert len(thresholds) == sample_image.GetDepth()
        
        # Check binary nature of output
        array = sitk.GetArrayFromImage(binary_image)
        assert np.array_equal(np.unique(array), np.array([0, 1]))

class TestVTKConverter:
    def test_convert_to_vtk_image(self):
        """Test conversion of binary volume to VTK image."""
        converter = VTKConverter()
        binary_volume = np.zeros((10, 10, 10), dtype=np.uint8)
        binary_volume[4:7, 4:7, 4:7] = 1
        
        spacing = (1.0, 1.0, 1.0)
        origin = (0.0, 0.0, 0.0)
        direction = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        
        vtk_image = converter.convert_to_vtk_image(binary_volume, spacing, origin, direction)
        
        assert isinstance(vtk_image, vtk.vtkImageData)
        assert vtk_image.GetDimensions() == binary_volume.shape
        assert vtk_image.GetSpacing() == spacing

    def test_extract_surface_mesh(self):
        """Test surface mesh extraction using marching cubes."""
        converter = VTKConverter()
        
        # Create a simple cube in vtkImageData
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(10, 10, 10)
        vtk_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        
        # Fill the center with 1s
        for z in range(3, 7):
            for y in range(3, 7):
                for x in range(3, 7):
                    vtk_image.SetScalarComponentFromDouble(x, y, z, 0, 1)
        
        mesh = converter.extract_surface_mesh(vtk_image)
        assert isinstance(mesh, vtk.vtkPolyData)
        assert mesh.GetNumberOfPoints() > 0
        assert mesh.GetNumberOfCells() > 0

    def test_smooth_mesh_laplacian(self):
        """Test Laplacian mesh smoothing."""
        converter = VTKConverter()
        
        # Create a simple mesh (cube)
        cube_source = vtk.vtkCubeSource()
        cube_source.Update()
        original_mesh = cube_source.GetOutput()
        
        smoothed_mesh = converter.smooth_mesh_laplacian(original_mesh)
        assert isinstance(smoothed_mesh, vtk.vtkPolyData)
        assert smoothed_mesh.GetNumberOfPoints() == original_mesh.GetNumberOfPoints()

class TestSurfaceMeshExtractor:
    @pytest.fixture
    def mock_mri_path(self, tmp_path):
        """Create a temporary MRI file for testing."""
        test_image = sitk.Image([20, 20, 20], sitk.sitkFloat32)
        test_file = tmp_path / "test_mri.nii.gz"
        sitk.WriteImage(test_image, str(test_file))
        return str(test_file)

    def test_extract_surface_mesh_with_otsu_global(self, mock_mri_path):
        """Test complete pipeline with global Otsu thresholding."""
        extractor = SurfaceMeshExtractor()
        mesh, mesh_smooth, binary_image, vtk_image, threshold = (
            extractor.extract_surface_mesh_with_otsu(mock_mri_path, otus_by_slice=False)
        )
        
        assert isinstance(mesh, vtk.vtkPolyData)
        assert isinstance(mesh_smooth, vtk.vtkPolyData)
        assert isinstance(binary_image, sitk.Image)
        assert isinstance(vtk_image, vtk.vtkImageData)
        assert isinstance(threshold, float)

    def test_extract_surface_mesh_with_otsu_slice(self, mock_mri_path):
        """Test complete pipeline with slice-by-slice Otsu thresholding."""
        extractor = SurfaceMeshExtractor()
        mesh, mesh_smooth, binary_image, vtk_image, thresholds = (
            extractor.extract_surface_mesh_with_otsu(mock_mri_path, otus_by_slice=True)
        )
        
        assert isinstance(mesh, vtk.vtkPolyData)
        assert isinstance(mesh_smooth, vtk.vtkPolyData)
        assert isinstance(binary_image, sitk.Image)
        assert isinstance(vtk_image, vtk.vtkImageData)
        assert isinstance(thresholds, list)

if __name__ == "__main__":
    pytest.main([__file__])