import os
import vtk
import numpy as np
from surface_mesh_extractor import SurfaceMeshExtractor
from render_and_landmarks import MeshRenderer, LandmarkDetector
from icp import ICPRegistration

def load_stl(stl_path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    reader.Update()
    return reader.GetOutput()

def main():
    # Initialize components
    mesh_extractor = SurfaceMeshExtractor()
    renderer = MeshRenderer()
    landmark_detector = LandmarkDetector()
    
    # Process MRI volume
    mri_path = "/Users/kedar/Desktop/brain-data-kiri/IXI-T1/IXI002-Guys-0828-T1.nii.gz"
    mri_mesh, mri_mesh_smooth, _, _, _ = mesh_extractor.extract_surface_mesh_with_otsu(mri_path, True)
    
    # Generate views and detect landmarks on MRI mesh
    mri_angles = [(90, 10, -90), (10, 0, 0), (-20, 0, 0)]
    mri_rendered_images = renderer.render_mesh_at_angles(mri_mesh_smooth, mri_angles)
    mri_cameras = renderer.get_cameras_for_angles(mri_angles)
    _, mri_landmarks_3d = landmark_detector.detect_face_landmarks_3d(
        mri_rendered_images, mri_cameras, renderer.render_window
    )
    
    # Load and process STL model
    stl_path = "/Users/kedar/Desktop/brain-data-kiri/Mannequin_Scan_092324/3DModel.stl"
    stl_mesh = load_stl(stl_path)
    
    # Generate views and detect landmarks on STL mesh
    stl_angles = [(90, 10, -90), (10, 0, 0), (-20, 0, 0)]
    stl_rendered_images = renderer.render_mesh_at_angles(stl_mesh, stl_angles)
    stl_cameras = renderer.get_cameras_for_angles(stl_angles)
    _, stl_landmarks_3d = landmark_detector.detect_face_landmarks_3d(
        stl_rendered_images, stl_cameras, renderer.render_window
    )
    
    # Perform ICP registration
    icp = ICPRegistration(
        fixed_landmarks=mri_landmarks_3d,
        moving_landmarks=stl_landmarks_3d,
        fixed_image=mri_mesh
    )
    
    # Set up ICP with scaling
    transform, inlier_indices = icp.perform_icp_registration(
        outlier_rejection_threshold=10.0
    )
    
    # Calculate registration metrics
    metrics = icp.calculate_registration_metrics(
        transform, 
        inlier_indices,
        distance_threshold=2.0
    )
    
    print("Registration Metrics:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")