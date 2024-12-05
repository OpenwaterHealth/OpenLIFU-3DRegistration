# render_and_landmarks.py

import vtk
import cv2
import mediapipe as mp
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np


class MeshRenderer:
    def render_mesh_at_angles(self, mesh, angles, image_size=(512, 512), flip_image=True, skin_color=(0.91, 0.76, 0.65), debug=False):
        """Render the mesh at different angles with a skin tone color and return the images."""
        images = []
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.SetSize(image_size[0], image_size[1])
        render_window.AddRenderer(renderer)
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        mapper.ScalarVisibilityOff()

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(skin_color)
        renderer.AddActor(actor)

        for angle in angles:
            camera = renderer.GetActiveCamera()
            
            # Apply rotations around Y (Azimuth), X (Elevation), and Z (Roll) axes
            camera.Azimuth(angle[0])   # Rotate around Y-axis
            camera.Elevation(angle[1]) # Rotate around X-axis
            camera.Roll(angle[2])      # Rotate around Z-axis (Roll)

            renderer.ResetCamera()

            render_window.Render()
            window_to_image_filter = vtk.vtkWindowToImageFilter()
            window_to_image_filter.SetInput(render_window)
            window_to_image_filter.Update()

            vtk_image = window_to_image_filter.GetOutput()
            width, height, _ = vtk_image.GetDimensions()
            vtk_array = vtk_to_numpy(vtk_image.GetPointData().GetScalars())
            img = vtk_array.reshape((height, width, -1))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            if flip_image:
                img = cv2.flip(img, 0)
            images.append(img)

            if debug:
                cv2.imshow("Rendered Image", img)
                cv2.waitKey(0)

        return images

    def get_cameras_for_angles(self, angles, image_size=(512, 512)):
        """Returns a list of VTK cameras configured for each angle."""
        cameras = []
        
        # Set up basic rendering components
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.SetSize(image_size[0], image_size[1])
        render_window.AddRenderer(renderer)
        
        for angle in angles:
            camera = vtk.vtkCamera()
            # Apply rotations around Y (Azimuth), X (Elevation), and Z (Roll) axes
            camera.Azimuth(angle[0]) 
            camera.Elevation(angle[1])
            camera.Roll(angle[2])
            
            # Set standard camera parameters
            camera.SetPosition(0, 0, 1)  # Starting position
            camera.SetFocalPoint(0, 0, 0)  # Looking at origin
            camera.SetViewUp(0, 1, 0)  # Up direction
            camera.SetClippingRange(0.1, 1000)
            
            cameras.append(camera)
            
        self.render_window = render_window  # Store for later use
        return cameras

    def animate(self, obj, event):
        """Callback function for the VTK animation."""
        camera = self.renderer.GetActiveCamera()
        camera.Azimuth(1)  # Adjust rotation speed as needed
        self.render_window.Render()

    def start_animation(self):
        """Starts the continuous rendering animation."""
        animationScene = vtk.vtkAnimationScene()
        animationScene.SetModeToRealTime()
        animationScene.SetLoop(0)  # Loop indefinitely

        animationCue = vtk.vtkAnimationCue()
        animationCue.SetStartTime(0)
        animationCue.SetEndTime(10000)  # Long end time
        animationCue.SetFrameRate(30)  # Adjust as needed
        animationCue.SetAnimationTypeToFunction()
        animationCue.SetCallback(self.animate)

        animationScene.AddCue(animationCue)
        animationScene.Start()

class LandmarkDetector:
    def detect_face_landmarks(self, images):
        """Detect facial landmarks in the rendered images and return the image with the highest confidence."""
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        annotated_images = []
        all_landmarks_2d = []
        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

        highest_confidence = -1  # To track the highest confidence value
        best_image_index = -1  # To track the index of the image with the highest confidence
        best_annotated_image = None  # To store the annotated image with the highest confidence
        best_landmarks_2d = []  # To store the landmarks of the image with the highest confidence

        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.1) as face_mesh:
            for idx, image in enumerate(images):
                # Convert the BGR image to RGB before processing.
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # If landmarks are found, check the confidence score and update if it's the highest
                if results.multi_face_landmarks and results.multi_face_landmarks[0]:
                    face_landmarks = results.multi_face_landmarks[0]

                    # For now, the detection confidence is part of the face_mesh process
                    # In the future, you may want to modify the model to expose this more clearly
                    detection_confidence = results.multi_face_landmarks[0].landmark[0].visibility

                    # Update the best detection based on confidence score
                    if detection_confidence > highest_confidence:
                        highest_confidence = detection_confidence
                        best_image_index = idx

                        # Copy the current image for annotation
                        best_annotated_image = image.copy()

                        # Annotate the image with the landmarks
                        mp_drawing.draw_landmarks(
                            image=best_annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_TESSELATION,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            image=best_annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_CONTOURS,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                        mp_drawing.draw_landmarks(
                            image=best_annotated_image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACEMESH_IRISES,
                            landmark_drawing_spec=None,
                            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                        # Extract the 2D landmark positions
                        best_landmarks_2d = [(landmark.x * image.shape[1], landmark.y * image.shape[0]) for landmark in face_landmarks.landmark]

                    # Append the annotated image regardless of confidence to the list
                    annotated_images.append(best_annotated_image)

                else:
                    # If no landmarks are detected, append the original image unaltered
                    annotated_images.append(image)

        # Return the image index, annotated image, and 2D landmarks with the highest confidence
        return best_image_index, best_annotated_image, best_landmarks_2d

    def unproject_2d_to_3d(self, camera, renderer, screen_points, depth_value=0.5):
        """Unproject 2D screen points to 3D points in the volume's coordinate space."""
        unprojected_points = []

        for screen_point in screen_points:
            x, y = screen_point
            
            # Normalize screen coordinates to [-1, 1] range
            norm_x = 2.0 * x / renderer.GetSize()[0] - 1.0
            norm_y = 2.0 * y / renderer.GetSize()[1] - 1.0

            # Use vtk's method to convert screen to world coordinates
            world_coords = [0.0, 0.0, 0.0]
            camera.ViewToWorld(norm_x, norm_y, depth_value, world_coords)

            unprojected_points.append(world_coords[:3])

        return unprojected_points

    def render_landmarks_in_volume(self, renderer, landmarks_3d, landmark_color=(1, 0, 0)):
        """Render the 3D landmarks inside the volume using VTK."""
        for point in landmarks_3d:
            # Create a sphere or point at each landmark position
            sphere_source = vtk.vtkSphereSource()
            sphere_source.SetCenter(point[0], point[1], point[2])
            sphere_source.SetRadius(1.0)  # Adjust radius as needed

            # Create a mapper and actor for the sphere
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphere_source.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(landmark_color)

            # Add the actor to the renderer
            renderer.AddActor(actor)

        return renderer

    def triangulate_3d_points(self, landmarks_2d_multiple_views, cameras, render_window):
        """Triangulate 3D points from multiple 2D views with known camera parameters."""
        points_3d = []
        
        # For each corresponding set of 2D points
        for i in range(len(landmarks_2d_multiple_views[0])):
            # Collect the same landmark from all views
            view_points = []
            view_matrices = []
            
            for view_idx, landmarks in enumerate(landmarks_2d_multiple_views):
                if landmarks:  # Check if landmarks exist for this view
                    point_2d = landmarks[i]
                    camera = cameras[view_idx]
                    
                    # Get camera parameters
                    proj_matrix = vtk.vtkMatrix4x4()
                    camera.GetProjectionTransformMatrix(proj_matrix)
                    view_matrix = vtk.vtkMatrix4x4()
                    camera.GetViewTransformMatrix(view_matrix)
                    
                    # Combine view and projection matrices
                    combined_matrix = vtk.vtkMatrix4x4()
                    vtk.vtkMatrix4x4.Multiply4x4(proj_matrix, view_matrix, combined_matrix)
                    
                    view_points.append(point_2d)
                    view_matrices.append(combined_matrix)
            
            if len(view_points) >= 2:  # Need at least 2 views for triangulation
                # Use DLT (Direct Linear Transform) to triangulate the 3D point
                point_3d = self._triangulate_point(view_points, view_matrices)
                points_3d.append(point_3d)
        
        return points_3d

    def _triangulate_point(self, points_2d, projection_matrices):
        """Implement Direct Linear Transform (DLT) for 3D point triangulation."""
        A = []
        for (x, y), P in zip(points_2d, projection_matrices):
            A.append([
                x * P.GetElement(2, 0) - P.GetElement(0, 0),
                x * P.GetElement(2, 1) - P.GetElement(0, 1),
                x * P.GetElement(2, 2) - P.GetElement(0, 2)
            ])
            A.append([
                y * P.GetElement(2, 0) - P.GetElement(1, 0),
                y * P.GetElement(2, 1) - P.GetElement(1, 1),
                y * P.GetElement(2, 2) - P.GetElement(1, 2)
            ])
        
        # Solve using SVD
        U, S, Vt = np.linalg.svd(A)
        point_3d = Vt[-1, :3] / Vt[-1, 3]
        return point_3d

    def detect_face_landmarks_3d(self, images, cameras, render_window):
        """Detect facial landmarks in multiple views and reconstruct 3D positions."""
        landmarks_2d_all_views = []
        annotated_images = []
        best_views = []
        
        # Detect landmarks in each view
        for image in images:
            idx, annotated_image, landmarks_2d = self.detect_face_landmarks([image])
            if idx != -1:
                landmarks_2d_all_views.append(landmarks_2d)
                annotated_images.append(annotated_image)
                best_views.append(True)
            else:
                landmarks_2d_all_views.append(None)
                annotated_images.append(image)
                best_views.append(False)
        
        # Reconstruct 3D points if we have at least 2 views with detected landmarks
        if sum(best_views) >= 2:
            # Filter out views without landmarks
            valid_landmarks = [lm for lm in landmarks_2d_all_views if lm is not None]
            valid_cameras = [cam for cam, is_valid in zip(cameras, best_views) if is_valid]
            
            # Triangulate 3D points
            landmarks_3d = self.triangulate_3d_points(valid_landmarks, valid_cameras, render_window)
            return annotated_images, landmarks_3d
        
        return annotated_images, None