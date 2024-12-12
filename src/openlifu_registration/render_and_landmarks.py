# render_and_landmarks.py

import vtk
import cv2
import mediapipe as mp
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np


class MeshRenderer:
    def __init__(self):
        """Initialize VTK rendering pipeline components"""
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetOffScreenRendering(1)
        
        # Initialize camera
        self.camera = self.renderer.GetActiveCamera()
        
        # Initialize mapper and actor
        self.mapper = vtk.vtkPolyDataMapper()
        self.mesh_actor = vtk.vtkActor()
        self.mesh_actor.SetMapper(self.mapper)
        
        # Set default properties
        self.mesh_actor.GetProperty().SetColor(0.91, 0.76, 0.65)
        self.mesh_actor.GetProperty().SetAmbient(0.1)
        self.mesh_actor.GetProperty().SetDiffuse(0.9)
        self.mesh_actor.GetProperty().SetSpecular(0.1)
        
        # Set window size
        self.render_window.SetSize(512, 512)

    def _ensure_pipeline_setup(self):
        """Ensure rendering pipeline is properly set up"""
        if not self.renderer.HasViewProp(self.mesh_actor):
            self.renderer.AddActor(self.mesh_actor)

    def _get_screenshot(self):
        """Capture the current render window as a numpy array"""
        window_to_image = vtk.vtkWindowToImageFilter()
        window_to_image.SetInput(self.render_window)
        window_to_image.Update()

        vtk_image = window_to_image.GetOutput()
        width, height, _ = vtk_image.GetDimensions()
        vtk_array = vtk_image.GetPointData().GetScalars()
        components = vtk_array.GetNumberOfComponents()
        
        numpy_array = vtk_to_numpy(vtk_array).reshape(height, width, components)
        numpy_array = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
        
        return numpy_array

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
    
    def render_mesh_and_get_camera_params(self, mesh, angles, image_size=(512, 512), flip_image=True, skin_color=(0.91, 0.76, 0.65), debug=False):
        """
        Render the mesh at different angles and return both the rendered images and camera parameters.
        
        Args:
            mesh: VTK mesh to render
            angles: List of (azimuth, elevation, roll) angles in degrees
            image_size: Tuple of (width, height) for rendering
            flip_image: Whether to flip the rendered image vertically
            skin_color: RGB tuple for mesh color
            debug: Whether to display rendered images
            
        Returns:
            tuple: (list of rendered images, list of camera parameter dictionaries)
        """
        images = []
        camera_params = []
        
        # Set up rendering pipeline
        renderer = vtk.vtkRenderer()
        render_window = vtk.vtkRenderWindow()
        render_window.SetOffScreenRendering(1)
        render_window.SetSize(image_size[0], image_size[1])
        render_window.AddRenderer(renderer)
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)

        # Set up mesh actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(mesh)
        mapper.ScalarVisibilityOff()
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(skin_color)
        renderer.AddActor(actor)

        for angle in angles:
            camera = renderer.GetActiveCamera()
            
            # Reset camera to initial position
            camera.SetPosition(0, 0, 1)
            camera.SetFocalPoint(0, 0, 0)
            camera.SetViewUp(0, 1, 0)
            
            # Apply rotations
            camera.Azimuth(angle[0])
            camera.Elevation(angle[1])
            camera.Roll(angle[2])
            
            # Reset camera to fit mesh in view
            renderer.ResetCamera()
            
            # Capture camera parameters after reset
            params = {
                'position': camera.GetPosition(),
                'focal_point': camera.GetFocalPoint(),
                'view_up': camera.GetViewUp(),
                'distance': camera.GetDistance(),
                'view_angle': camera.GetViewAngle(),
                'clipping_range': camera.GetClippingRange(),
                'parallel_scale': camera.GetParallelScale(),
                'parallel_projection': camera.GetParallelProjection(),
                'thickness': camera.GetThickness(),
                'window_center': camera.GetWindowCenter(),
                'applied_angles': angle  # Store the original angles used
            }
            camera_params.append(params)
            
            # Render and capture image
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

            return images, camera_params

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
    def detect_face_landmarks(self, image):
        """
        Detect facial landmarks in a single rendered image.
        
        Args:
            image: Single image array
            
        Returns:
            tuple: (annotated_image, landmarks_2d) where landmarks_2d is a list of (x,y) coordinates
        """
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        annotated_image = image.copy()
        landmarks_2d = []
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.1
        ) as face_mesh:
            # Convert to RGB
            results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.multi_face_landmarks and results.multi_face_landmarks[0]:
                face_landmarks = results.multi_face_landmarks[0]
                
                # Draw the landmarks
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,  
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                    image=annotated_image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())
                
                # Extract 2D landmark positions
                landmarks_2d = [(landmark.x * image.shape[1], landmark.y * image.shape[0]) 
                              for landmark in face_landmarks.landmark]
                
                return annotated_image, landmarks_2d
                
            return image, None

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
            annotated_image, landmarks_2d = self.detect_face_landmarks(image)
            if landmarks_2d is not None:
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