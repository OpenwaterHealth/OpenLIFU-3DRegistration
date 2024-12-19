import os
import vtk, cv2
import numpy as np
from surface_mesh_extractor import SurfaceMeshExtractor
from render_and_landmarks import MeshRenderer, LandmarkDetector
from icp import ICPRegistration, ProcrustesAligner
import h5py

DEBUG_IMAGES = False

def load_obj(obj_path):
    reader = vtk.vtkOBJReader()
    reader.SetFileName(obj_path)
    reader.Update()
    return reader.GetOutput()

def load_stl(stl_path):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_path)
    reader.Update()
    return reader.GetOutput()

def load_transform(transform_path):
    try:
        with h5py.File(transform_path, 'r') as f:
            # Verify required datasets exist
            if 'TransformGroup/0/TransformParameters' not in f or \
               'TransformGroup/0/TransformFixedParameters' not in f:
                raise KeyError("Required transform datasets not found in HDF5 file")

            params = f['TransformGroup/0/TransformParameters'][()]
            fixed_params = f['TransformGroup/0/TransformFixedParameters'][()]

            # Initialize 4x4 homogeneous transformation matrix
            transform = np.zeros((4, 4))
            transform[:3, :] = params.reshape(3, 4)
            transform[3, :3] = fixed_params
            transform[3, 3] = 1

            return transform

    except (OSError, KeyError) as e:
        raise RuntimeError(f"Failed to load transducer transform: {str(e)}")

def main():
    # Initialize components
    renderer = MeshRenderer()
    landmark_detector = LandmarkDetector()
    
    # Load MRI mesh
    mri_path = "/Users/kedar/Desktop/brain-data-kiri/RegistrationExampleData/Input_Extracted_MRI_Surface/T1.surface.obj"
    mri_mesh = load_obj(mri_path)

    # Load and process physical scan
    phMesh_path = "/Users/kedar/Desktop/brain-data-kiri/RegistrationExampleData/_Input_Photogrammetry_raw_meshroom/texturedMesh.obj"
    phMesh_mesh = load_obj(phMesh_path)
    
    try:
        # Generate views and detect landmarks on MRI mesh
        mri_angles = [(0, -80, 0), (20, -80, 0), (0, -80, 10), (0, -80, -20), (0, -80, 20)]
        mri_rendered_images, cameras = renderer.render_mesh_and_get_camera_params(mri_mesh, mri_angles)
        if DEBUG_IMAGES:
            for i, img in enumerate(mri_rendered_images):
                cv2.imwrite(f"rendered_mri_image_new{i}.png", img)
        mri_annotated_images, mri_landmarks_3d, mri_landmark_errors = landmark_detector.detect_face_landmarks_3d(
                    mri_rendered_images, cameras, renderer.render_window)
        if DEBUG_IMAGES:
            for i, img in enumerate(mri_annotated_images):
                cv2.imwrite(f"annotated_mri_image_new{i}.png", img)
        
        #Initialize the pipeline objects for the physical scan
        renderer_phone = MeshRenderer()
        landmark_detector_phone = LandmarkDetector()
        
        # Generate views and detect landmarks on phone scan mesh
        phMesh_angles = [(0, -75, 0), (20, -80, 0), (0, -80, 10)]
        phMesh_rendered_images, cameras = renderer_phone.render_mesh_and_get_camera_params(phMesh_mesh, phMesh_angles)
        if DEBUG_IMAGES:
            for i, img in enumerate(phMesh_rendered_images):
                cv2.imwrite(f"rendered_phone_image_new{i}.png", img)
        pMesh_annotated_images, phMesh_landmarks_3d, phMesh_landmarks_errors = landmark_detector_phone.detect_face_landmarks_3d(
            phMesh_rendered_images, cameras, renderer_phone.render_window
        )
        if DEBUG_IMAGES:
            for i, img in enumerate(pMesh_annotated_images):
                cv2.imwrite(f"annotated_phone_image_new{i}.png", img)
        
        # After detecting landmarks
        procrustes = ProcrustesAligner()
        procrustes.align(phMesh_landmarks_3d, mri_landmarks_3d)
        initial_transform = procrustes.get_transformation_matrix()
        print("Initial Procrustes Alignment Matrix:")
        print(initial_transform)

        # Perform ICP registration between MRI and physical scan
        icp = ICPRegistration(mode='similarity')
        phoneScan_transform, registration_error = icp.register(phMesh_mesh, mri_mesh, initial_transform=initial_transform)
        print("Phone to MRI ICP Registration Matrix:")
        print(phoneScan_transform)

    except Exception as e:
        print(f"Error: {e}")
        print("Using precomputed transformation matrix")
        phoneScan_transform_path = '/Users/kedar/Desktop/brain-data-kiri/RegistrationExampleData/Output_Scan_to_MRI/Scan_To_MRI_Fiducial.h5'
        phoneScan_transform = load_transform(phoneScan_transform_path)
        
    #Apply the transformation to the physical scan to align it with the MRI
    phone_transform_filter = vtk.vtkTransformPolyDataFilter()
    phone_transform_filter.SetInputData(phMesh_mesh)
    phone_transform = vtk.vtkTransform()
    phone_transform.SetMatrix(phoneScan_transform.ravel())
    phone_transform_filter.SetTransform(phone_transform)
    phone_transform_filter.Update()
    transformed_phone_mesh = phone_transform_filter.GetOutput()

    # Load transducer geometry
    transducer_path = "/Users/kedar/Desktop/brain-data-kiri/RegistrationExampleData/Input_Transducer_Geometry/transducer.surf.obj"
    transducer_mesh = load_obj(transducer_path)

    # Load transducer estimated position in MRI space
    transducer_transform_path = "/Users/kedar/Desktop/brain-data-kiri/RegistrationExampleData/Input_Virtual_Fit_Transforms/transducer-matrix_search0.h5"
    initial_transducer_transform = load_transform(transducer_transform_path)

    # Apply the transformation to the transducer to align it with the MRI
    transducer_transform_filter = vtk.vtkTransformPolyDataFilter()
    transducer_transform_filter.SetInputData(transducer_mesh)
    transducer_transform = vtk.vtkTransform()
    transducer_transform.SetMatrix(initial_transducer_transform.ravel())
    transducer_transform_filter.SetTransform(transducer_transform)
    transducer_transform_filter.Update()
    initial_transducer_mesh = transducer_transform_filter.GetOutput()

    # Register the transducer to the physical scan in MRI space using ICP with rigid transform
    icp = ICPRegistration(mode='rigid')
    final_transducer_transform, registration_error = icp.register(initial_transducer_mesh, transformed_phone_mesh)
    print("Transducer to Phone ICP Registration Matrix:")
    print(final_transducer_transform)

if __name__ == "__main__":
    main()