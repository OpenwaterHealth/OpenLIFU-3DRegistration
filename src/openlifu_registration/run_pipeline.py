# run_pipeline.py

import os, cv2, random, numpy as np, logging
from surface_mesh_extractor import SurfaceMeshExtractor, save_mesh, save_image
from render_and_landmarks import MeshRenderer, LandmarkDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEBUG_IMAGES = True

def generate_random_angles(num_angles=30):
    """Generates a list of num_angles tuples, each containing 3 random angles 
    between 0 and 180 degrees for the X, Y, and Z axes.
    """
    angles = [(90, 10, -90), (10, 0, 0), (10, 0, 0), (-20, 0, 0), (-10, 0, 0)]
    return angles

def process_mri_volume(mri_volume_path, output_folder):
    """Processes a single MRI volume to extract the surface mesh, render images, and detect landmarks."""
    try:
        # Initialize the pipeline objects
        mesh_extractor = SurfaceMeshExtractor()
        renderer = MeshRenderer()
        landmark_detector = LandmarkDetector()

        logging.info(f"Processing: {mri_volume_path}")

        # Step 1: Extract surface mesh
        mesh, mesh_smooth, binary_image, vtk_image, threshold = mesh_extractor.extract_surface_mesh_with_otsu(mri_volume_path, True)
        logging.info(f"Estimated Otsu threshold: {threshold}")

        attempts = 5
        while attempts > 0:
            # Generate multiple views for 3D reconstruction
            angles = generate_random_angles(num_angles=3)  # Use 3 views
            rendered_images, cameras = renderer.render_mesh_with_cameras(mesh_smooth, angles)
            if DEBUG_IMAGES:
                for i, img in enumerate(rendered_images):
                    cv2.imwrite(f"rendered_image_new{i}.png", img)
            stl_rendered_images = renderer.render_mesh_at_angles(mesh_smooth, angles)
            if DEBUG_IMAGES:
                for i, img in enumerate(stl_rendered_images):
                    cv2.imwrite(f"rendered_image_old{i}.png", img)

            # Detect landmarks and reconstruct 3D positions
            annotated_images, landmarks_3d = landmark_detector.detect_face_landmarks_3d(
                rendered_images, cameras, renderer.render_window)

            if landmarks_3d is not None:
                # Save the annotated images and 3D landmarks
                filename = os.path.splitext(os.path.basename(mri_volume_path))[0]
                for i, img in enumerate(annotated_images):
                    output_image_path = os.path.join(
                        output_folder, 
                        f"{filename}_view{i}_az{angles[i][0]:.1f}_el{angles[i][1]:.1f}_roll{angles[i][2]:.1f}.png"
                    )
                    cv2.imwrite(output_image_path, img)

                # Save 3D landmarks to file
                landmark_file = os.path.join(output_folder, f"{filename}_landmarks_3d.npy")
                np.save(landmark_file, landmarks_3d)
                break

            attempts -= 1

        if landmarks_3d is None:
            logging.warning("Failed to detect face in sufficient views for 3D reconstruction")

    except Exception as e:
        logging.error(f"Error processing {mri_volume_path}: {e}")

def main(folder_path, output_folder):
    """Main function to process all MRI volumes in the specified folder."""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".nii.gz"):
                mri_volume_path = os.path.join(root, file)
                process_mri_volume(mri_volume_path, output_folder)

if __name__ == "__main__":
    folder_path = "/Users/kedar/Desktop/brain-data-kiri/IXI-T1/"  # Replace with your MRI volume folder path
    output_folder = "/path/to/output/folder"  # Replace with your desired output folder path
    main(folder_path, output_folder)
