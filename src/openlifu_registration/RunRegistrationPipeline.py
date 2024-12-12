import os
import vtk
import numpy as np
from surface_mesh_extractor import SurfaceMeshExtractor
from render_and_landmarks import MeshRenderer, LandmarkDetector
from icp import ICPRegistration

def load_obj(obj_path):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_path)
    reader.Update()
    return reader.GetOutput()

def main():
    # Initialize components
    renderer = MeshRenderer()
    landmark_detector = LandmarkDetector()
    
    # Load MRI mesh
    mri_path = "/Users/kedar/Desktop/brain-data-kiri/RegistrationExampleData/Input_Extracted_MRI_Surface/T1.surface.obj"
    mri_mesh = load_obj(mri_path)
    
    # Generate views and detect landmarks on MRI mesh
    mri_angles = [(90, 10, -90), (10, 0, 0), (-20, 0, 0)]
    mri_rendered_images = renderer.render_mesh_at_angles(mri_mesh, mri_angles)
    mri_cameras = renderer.get_cameras_for_angles(mri_angles)
    _, mri_landmarks_3d = landmark_detector.detect_face_landmarks_3d(
        mri_rendered_images, mri_cameras, renderer.render_window
    )
    
    # Load and process physical scan
    stl_path = "/Users/kedar/Desktop/brain-data-kiri/RegistrationExampleData/_Input_Photogrammetry_raw_meshroom/texturedMesh.obj"
    stl_mesh = load_obj(stl_path)
    
    # Generate views and detect landmarks on STL mesh
    stl_angles = [(90, 10, -90), (10, 0, 0), (-20, 0, 0)]
    stl_rendered_images = renderer.render_mesh_at_angles(stl_mesh, stl_angles)
    stl_cameras = renderer.get_cameras_for_angles(stl_angles)
    _, stl_landmarks_3d = landmark_detector.detect_face_landmarks_3d(
        stl_rendered_images, stl_cameras, renderer.render_window
    )
    
    # Perform ICP registration between MRI and physical scan
    icp = ICPRegistration(
        fixed_landmarks=mri_landmarks_3d,
        moving_landmarks=stl_landmarks_3d,
        fixed_image=mri_mesh
    )
    transform, inlier_indices = icp.perform_icp_registration(outlier_rejection_threshold=10.0)
    metrics = icp.calculate_registration_metrics(transform, inlier_indices, distance_threshold=2.0)

    print("Registration Metrics for MRI and Physical Scan:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")

    # Load transducer geometry
    transducer_path = "/Users/kedar/Desktop/brain-data-kiri/RegistrationExampleData/Input_Transducer_Geometry/transducer.surf.obj"
    transducer_mesh = load_obj(transducer_path)

    # Generate views and detect landmarks on transducer mesh
    transducer_angles = [(90, 10, -90), (10, 0, 0), (-20, 0, 0)]
    transducer_rendered_images = renderer.render_mesh_at_angles(transducer_mesh, transducer_angles)
    transducer_cameras = renderer.get_cameras_for_angles(transducer_angles)
    _, transducer_landmarks_3d = landmark_detector.detect_face_landmarks_3d(
        transducer_rendered_images, transducer_cameras, renderer.render_window
    )

    # Perform ICP registration between transducer and physical scan
    icp_transducer = ICPRegistration(
        fixed_landmarks=stl_landmarks_3d,
        moving_landmarks=transducer_landmarks_3d,
        fixed_image=stl_mesh
    )
    transform_transducer, inlier_indices_transducer = icp_transducer.perform_icp_registration(outlier_rejection_threshold=10.0)
    metrics_transducer = icp_transducer.calculate_registration_metrics(transform_transducer, inlier_indices_transducer, distance_threshold=2.0)

    print("Registration Metrics for Transducer and Physical Scan:")
    for metric_name, value in metrics_transducer.items():
        print(f"{metric_name}: {value}")