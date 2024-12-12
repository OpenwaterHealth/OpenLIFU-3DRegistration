import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np

SAVE_DEBUG_IMAGES = True
def save_image(image, filename):
    """Saves the image to a file."""
    sitk.WriteImage(image, filename)
def save_mesh(mesh, filename):
    """Saves the mesh in STL format."""
    writer = vtk.vtkSTLWriter()
    writer.SetFileName(filename)
    writer.SetInputData(mesh)
    writer.Write()

class MRIProcessor:
    def read_mri_volume(self, mri_volume_path):
        """Reads the MRI volume from a file."""
        return sitk.ReadImage(mri_volume_path)

    def apply_otsu_threshold_global(self, image):
        """Applies Otsu thresholding to the MRI volume and returns the binary image and threshold value."""
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetOutsideValue(0)
        otsu_filter.SetInsideValue(1)
        binary_image = otsu_filter.Execute(image)
        threshold = otsu_filter.GetThreshold()
        return binary_image, threshold
    
    def apply_otsu_threshold_slice_by_slice(self, image):
        """Applies Otsu thresholding slice by slice and adjusts thresholds outside 1 standard deviation from the mean."""
        image_array = sitk.GetArrayFromImage(image)
        z_dim = image.GetDepth()
        thresholds = [self._apply_otsu_to_slice(image_array[z, :, :]) for z in range(z_dim)]
        mean_threshold = int(np.mean(thresholds)+0.5)
        std_threshold = int(np.std(thresholds)+0.5)
        adjusted_thresholds = [mean_threshold if abs(t - mean_threshold) > std_threshold else t for t in thresholds]
        binary_image_array = np.array([image_array[z, :, :] > adjusted_thresholds[z] for z in range(z_dim)], dtype=np.uint8)
        binary_image = sitk.GetImageFromArray(binary_image_array)
        binary_image.CopyInformation(image)
        if SAVE_DEBUG_IMAGES:
            save_image(binary_image, "binary_image.nii.gz")
        return binary_image, adjusted_thresholds

    def _apply_otsu_to_slice(self, slice_image):
        """Applies Otsu thresholding to a single slice."""
        slice_image = sitk.GetImageFromArray(slice_image)
        otsu_filter = sitk.OtsuThresholdImageFilter()
        otsu_filter.SetOutsideValue(0)
        otsu_filter.SetInsideValue(1)
        otsu_filter.Execute(slice_image)
        return otsu_filter.GetThreshold()

class ITKtoVTKConverter:
    def convert_to_vtk_image(self, binary_volume, image_spacing, image_origin, image_direction):
        """Converts a binary volume (NumPy array) to a VTK image."""
        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(binary_volume.shape)
        vtk_image.SetSpacing(image_spacing)
        vtk_image.SetOrigin(image_origin)
        vtk_image.SetDirectionMatrix(image_direction)
        flat_binary_volume = binary_volume.flatten(order='F')
        vtk_array = numpy_support.numpy_to_vtk(num_array=flat_binary_volume, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_image.GetPointData().SetScalars(vtk_array)
        return vtk_image

    def extract_surface_mesh(self, vtk_image):
        """Extracts the surface mesh from a VTK image using the marching cubes algorithm."""
        marching_cubes = vtk.vtkMarchingCubes()
        marching_cubes.SetInputData(vtk_image)
        marching_cubes.SetValue(0, 0.5)  # Assuming binary volume with 0 and 1 values
        marching_cubes.Update()
        if SAVE_DEBUG_IMAGES:
            save_mesh(marching_cubes.GetOutput(), "marching_cubes_mesh.stl")
        return marching_cubes.GetOutput()
    
    def smooth_mesh_laplacian(self, mesh, iterations=50, relaxation_factor=0.5, feature_edge_smoothing=False, boundary_smoothing=False):
        """Smooths the surface mesh using Laplacian smoothing."""
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputData(mesh)
        smoother.SetNumberOfIterations(iterations)
        smoother.SetRelaxationFactor(relaxation_factor)
        smoother.SetFeatureEdgeSmoothing(feature_edge_smoothing)
        smoother.SetBoundarySmoothing(boundary_smoothing)
        smoother.Update()
        if SAVE_DEBUG_IMAGES:
            save_mesh(smoother.GetOutput(), "smoothed_mesh.stl")
        return smoother.GetOutput()

class SurfaceMeshExtractor:
    def __init__(self):
        self.processor = MRIProcessor()
        self.converter = ITKtoVTKConverter()

    def extract_surface_mesh_with_otsu(self, mri_volume_path, otsu_by_slice=False):
        """Extracts the surface mesh from an MRI volume using Otsu thresholding."""
        image = self.processor.read_mri_volume(mri_volume_path)
        binary_image, threshold = self._apply_otsu_threshold(image, otsu_by_slice)
        binary_volume = sitk.GetArrayFromImage(binary_image).astype(np.uint8)
        vtk_image = self.converter.convert_to_vtk_image(binary_volume, image.GetSpacing(), image.GetOrigin(), image.GetDirection())
        mesh = self.converter.extract_surface_mesh(vtk_image)
        mesh_smooth = self.converter.smooth_mesh_laplacian(mesh)
        return mesh, mesh_smooth, binary_image, vtk_image, threshold

    def _apply_otsu_threshold(self, image, otsu_by_slice):
        """Applies Otsu thresholding to the MRI volume."""
        if otsu_by_slice:
            return self.processor.apply_otsu_threshold_slice_by_slice(image)
        else:
            return self.processor.apply_otsu_threshold_global(image)

if __name__ == "__main__":
    extractor = SurfaceMeshExtractor()
    mri_volume_path = "/Users/kedar/Desktop/brain-data-kiri/IXI-T1/IXI211-HH-1568-T1.nii.gz"  # Replace with your MRI volume path

    # Test with global Otsu thresholding
    mesh_global, mesh_global_smooth, binary_image_global, vtk_image_global, threshold_global = extractor.extract_surface_mesh_with_otsu(mri_volume_path)
    print(f"Global Otsu threshold: {threshold_global}")
    save_mesh(mesh_global, "mesh_global.stl")
    save_mesh(mesh_global_smooth, "mesh_global_smooth.stl")

    # Test with slice-by-slice Otsu thresholding
    mesh_slice, mesh_slice_smooth, binary_image_slice, vtk_image_slice, thresholds_slice = extractor.extract_surface_mesh_with_otsu(mri_volume_path, otsu_by_slice=True)
    print(f"Slice-by-slice Otsu thresholds: {thresholds_slice}")
    save_mesh(mesh_slice, "mesh_slice.stl")
    save_mesh(mesh_slice_smooth, "mesh_slice_smooth.stl")
