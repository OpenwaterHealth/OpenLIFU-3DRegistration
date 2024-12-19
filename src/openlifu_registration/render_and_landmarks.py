# render_and_landmarks.py

import vtk
import cv2
import mediapipe as mp
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
from scipy.optimize import least_squares

DEBUG_IMAGES = False
DEBUG_MESH = False

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

    def get_camera_matrix(self, image_size):
        """Calculate camera intrinsic matrix from VTK parameters"""
        width, height = image_size
        view_angle = self.camera.GetViewAngle()
        
        # Calculate focal length from field of view
        focal_length = (width/2) / np.tan(np.deg2rad(view_angle/2))
        
        # Construct camera matrix
        K = np.array([
            [focal_length, 0, width/2],
            [0, focal_length, height/2],
            [0, 0, 1]
        ])
        return K

    def get_extrinsic_matrix(self, camera_params):
        """Calculate camera extrinsic matrix from VTK camera parameters"""
        position = np.array(camera_params['position'])
        focal_point = np.array(camera_params['focal_point'])
        view_up = np.array(camera_params['view_up'])
        
        # Calculate camera axes
        n = position - focal_point  # Camera z-axis
        n = n / np.linalg.norm(n)
        u = np.cross(view_up, n)   # Camera x-axis
        u = u / np.linalg.norm(u)
        v = np.cross(n, u)         # Camera y-axis
        
        # Construct rotation matrix
        R = np.vstack((u, v, n)).T
        
        # Construct translation vector
        t = -R @ position
        
        # Construct extrinsic matrix [R|t]
        extrinsic = np.hstack((R, t.reshape(3,1)))
        return extrinsic
    
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
            
            # First get the camera parameters
            temp_params = {
                'position': camera.GetPosition(),
                'focal_point': camera.GetFocalPoint(),
                'view_up': camera.GetViewUp()
            }
            
            # Now create full params dictionary with matrices
            params = {
                'position': temp_params['position'],
                'focal_point': temp_params['focal_point'],
                'view_up': temp_params['view_up'],
                'distance': camera.GetDistance(),
                'view_angle': camera.GetViewAngle(),
                'clipping_range': camera.GetClippingRange(),
                'parallel_scale': camera.GetParallelScale(),
                'parallel_projection': camera.GetParallelProjection(),
                'thickness': camera.GetThickness(),
                'window_center': camera.GetWindowCenter(),
                'applied_angles': angle,
                'camera_matrix': self.get_camera_matrix(image_size),
                'extrinsic_matrix': self.get_extrinsic_matrix(temp_params)
            }
            camera_params.append(params)
            
            # Render and capture image
            render_window.Render()
            # Debug: show VTK window
            if DEBUG_MESH:
                render_window.SetOffScreenRendering(0)  # Enable on-screen rendering
                render_window.Render()
                render_window_interactor = vtk.vtkRenderWindowInteractor()
                render_window_interactor.SetRenderWindow(render_window)
                render_window_interactor.Initialize()
                render_window_interactor.Start()
                render_window.SetOffScreenRendering(1)  # Re-enable off-screen rendering
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

class Triangulator:
    def __init__(self):
        self.min_views = 2

    def get_projection_matrix(self, camera_params):
        """Compute projection matrix P = K[R|t]"""
        K = camera_params['camera_matrix']
        RT = camera_params['extrinsic_matrix']
        return np.dot(K, RT)

    def triangulate_point(self, point_2d_list, projection_matrices):
        """Linear triangulation using DLT"""
        A = []
        for point_2d, P in zip(point_2d_list, projection_matrices):
            x, y = point_2d
            A.append([x*P[2] - P[0]])
            A.append([y*P[2] - P[1]])
        
        A = np.array(A).reshape(-1, 4)
        _, _, Vt = np.linalg.svd(A)
        point_3d = Vt[-1]
        return point_3d[:3] / point_3d[3]

    def reprojection_error(self, point_3d, point_2d_list, projection_matrices):
        """Compute reprojection error"""
        errors = []
        point_3d_homog = np.append(point_3d, 1)
        
        for point_2d, P in zip(point_2d_list, projection_matrices):
            point_proj = np.dot(P, point_3d_homog)
            point_proj = point_proj[:2] / point_proj[2]
            error = np.linalg.norm(point_2d - point_proj)
            errors.append(error)
        
        return np.array(errors)

    def optimize_point(self, point_3d_init, point_2d_list, projection_matrices):
        """Bundle adjustment for single point"""
        def objective(point_3d):
            return self.reprojection_error(point_3d, point_2d_list, projection_matrices)
        
        result = least_squares(objective, point_3d_init)
        return result.x

    def triangulate_landmarks(self, points_2d_views, camera_params):
        """Triangulate all landmarks across views"""
        projection_matrices = [self.get_projection_matrix(cam) for cam in camera_params]
        
        num_landmarks = len(points_2d_views[0])
        landmarks_3d = []
        landmark_errors = []  # Reset errors for new triangulation

        for i in range(num_landmarks):
            point_2d_list = [view[i] for view in points_2d_views]
            
            # Initial triangulation
            point_3d = self.triangulate_point(point_2d_list, projection_matrices)
            
            # Optimize
            point_3d_refined = self.optimize_point(point_3d, point_2d_list, projection_matrices)
            
            # Store point and its reprojection error
            errors = self.reprojection_error(point_3d_refined, point_2d_list, projection_matrices)
            landmarks_3d.append(point_3d_refined)
            landmark_errors.append(np.median(errors))
            
        return np.array(landmarks_3d), np.array(landmark_errors)

class LandmarkDetector:
    def __init__(self):
        self.triangulator = Triangulator()

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
            min_detection_confidence=0.01
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
        
        if DEBUG_IMAGES is True:
            for i, img in enumerate(annotated_images):
                cv2.imwrite(f"annotated_image_{i}.png", img)

        # Reconstruct 3D points if we have at least 2 views with detected landmarks
        landmarks_3d = None
        landmark_errors = None
        if sum(best_views) >= 2:
            # Filter out views without landmarks
            valid_landmarks = [lm for lm, is_valid in zip(landmarks_2d_all_views, best_views) if is_valid]
            valid_camera_params = [cam for cam, is_valid in zip(cameras, best_views) if is_valid]
            
            # Reconstruct 3D landmarks
            landmarks_3d, landmark_errors = self.triangulator.triangulate_landmarks(valid_landmarks, valid_camera_params)
        
        return annotated_images, landmarks_3d, landmark_errors