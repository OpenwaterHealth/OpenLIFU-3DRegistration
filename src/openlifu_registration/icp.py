import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np

class ProcrustesAligner:
    def __init__(self):
        self.scale = None
        self.rotation = None
        self.translation = None
        self.rmse = None

    def align(self, source, target):
        """
        Compute Procrustes alignment between source and target points.
        
        Args:
            source: (N,3) array of source points
            target: (N,3) array of target points
        """
        # Convert to numpy arrays and ensure correct shape
        source = np.asarray(source)
        target = np.asarray(target)
        
        # Get centroids
        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(target, axis=0)
        
        # Center both point sets
        source_centered = source - source_centroid
        target_centered = target - target_centroid
        
        # Compute scale
        source_norm = np.linalg.norm(source_centered, 'fro')
        target_norm = np.linalg.norm(target_centered, 'fro')
        self.scale = target_norm / source_norm
        
        # Get optimal rotation
        H = target_centered.T @ source_centered
        U, _, Vt = np.linalg.svd(H)
        
        # Handle reflection case
        V = Vt.T
        det = np.linalg.det(U @ Vt)
        if det < 0:
            V[:, -1] *= -1
        
        self.rotation = U @ V.T
        self.translation = target_centroid - self.scale * (self.rotation @ source_centroid)
        
        # Compute alignment error
        transformed = self.transform(source)
        self.rmse = np.sqrt(np.mean(np.sum((transformed - target)**2, axis=1)))
        
        return self

    def transform(self, points):
        """Apply transformation to points"""
        return self.scale * (self.rotation @ points.T).T + self.translation

    def get_transformation_matrix(self):
        """Return 4x4 homogeneous transformation matrix"""
        matrix = np.eye(4)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix

class ICPRegistration:
    def __init__(self, mode='similarity'):
        """
        Initialize ICP registration.
        Args:
            mode (str): 'similarity' or 'rigid'
        """
        self.mode = mode
        self.initial_transform = None
        self.final_transform = None
        self.registration_error = None
        
    def register(self, source_mesh, target_mesh, initial_transform=None, 
                max_iterations=100, tolerance=0.001):
        """
        Perform ICP registration.
        Source and target meshes should be vtkPolyData objects.
        Source is the moving mesh, target is the fixed mesh.
        """
        # Set up ICP transform
        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(source_mesh)
        icp.SetTarget(target_mesh)
        
        # Configure transform type
        if self.mode == 'similarity':
            icp.GetLandmarkTransform().SetModeToSimilarity()
        elif self.mode == 'rigid':
            icp.GetLandmarkTransform().SetModeToRigidBody()
        else:
            raise ValueError("Invalid mode. Choose 'similarity' or 'rigid'")
        
        # Set ICP parameters
        icp.SetMaximumNumberOfIterations(max_iterations)
        icp.SetMaximumMeanDistance(tolerance)
        icp.SetCheckMeanDistance(1)
        
        # Apply initial transform if provided
        if initial_transform is not None:
            self.initial_transform = initial_transform
            initial_vtk_transform = vtk.vtkTransform()
            initial_vtk_transform.SetMatrix(initial_transform.ravel())
            icp.SetInitialTransform(initial_vtk_transform)
        
        # Run registration
        icp.Modified()
        icp.Update()
        
        # Store results
        self.final_transform = np.array(icp.GetMatrix().GetData())
        self.registration_error = icp.GetMeanDistance()
        
        return self.final_transform, self.registration_error
    
    def transform_mesh(self, mesh):
        """
        Apply final transformation to a mesh.
        """
        if self.final_transform is None:
            raise RuntimeError("Registration must be performed first")
            
        transform = vtk.vtkTransform()
        transform.SetMatrix(self.final_transform.ravel())
        
        transformer = vtk.vtkTransformPolyDataFilter()
        transformer.SetTransform(transform)
        transformer.SetInputData(mesh)
        transformer.Update()
        
        return transformer.GetOutput()
    
    def transform_points(self, points):
        """
        Apply final transformation to points.
        """
        if self.final_transform is None:
            raise RuntimeError("Registration must be performed first")
            
        # Convert to homogeneous coordinates
        points_h = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed_points = (self.final_transform @ points_h.T).T
        
        return transformed_points[:, :3]

    def calculate_registration_metrics(self, transform, inlier_indices, distance_threshold=2.0):
        """
        Calculates registration metrics.

        Args:
            transform (vtk.vtkTransform): The transformation obtained from ICP.
            inlier_indices (list): List of inlier indices.
            distance_threshold (float): Threshold distance for considering points in metrics calculation.

        Returns:
            dict: A dictionary containing the calculated metrics.
        """
        metrics = {}

        # Get transformation matrix to extract scale
        matrix = vtk.vtkMatrix4x4()
        transform.GetMatrix(matrix)
        
        # Calculate scale factor from matrix
        scale_x = np.sqrt(matrix.GetElement(0,0)**2 + matrix.GetElement(0,1)**2 + matrix.GetElement(0,2)**2)
        scale_y = np.sqrt(matrix.GetElement(1,0)**2 + matrix.GetElement(1,1)**2 + matrix.GetElement(1,2)**2)
        scale_z = np.sqrt(matrix.GetElement(2,0)**2 + matrix.GetElement(2,1)**2 + matrix.GetElement(2,2)**2)
        
        metrics["Scale_X"] = scale_x
        metrics["Scale_Y"] = scale_y
        metrics["Scale_Z"] = scale_z

        # Apply the transformation to the moving landmarks
        transformed_points = vtk.vtkPoints()
        moving_points = vtk.vtkPoints()
        for i in inlier_indices:
            moving_points.InsertNextPoint(self.moving_landmarks[i])
        transform.TransformPoints(moving_points, transformed_points)

        # Convert to NumPy arrays for easier calculations
        fixed_points_np = np.array([self.fixed_landmarks[i] for i in inlier_indices])
        transformed_points_np = numpy_support.vtk_to_numpy(transformed_points.GetData())

        # Calculate distances between corresponding points
        distances = np.sqrt(np.sum((fixed_points_np - transformed_points_np)**2, axis=1))

        # Filter distances based on the threshold
        filtered_distances = distances[distances <= distance_threshold]

        # Calculate metrics
        if len(filtered_distances) > 0:
            metrics["RMSE"] = np.sqrt(np.mean(filtered_distances**2))
            metrics["MAE"] = np.mean(filtered_distances)

            # Hausdorff Distance
            hausdorff_distance_filter = vtk.vtkHausdorffDistancePointSetFilter()
            hausdorff_distance_filter.SetInputData(0, self.moving_polydata)
            hausdorff_distance_filter.SetInputData(1, self.fixed_polydata)
            hausdorff_distance_filter.Update()
            metrics["Hausdorff Distance"] = hausdorff_distance_filter.GetHausdorffDistance()

            # Chamfer Distance (approximation)
            metrics["Chamfer Distance"] = (np.sum(np.min(distances, axis=0)) + np.sum(np.min(distances, axis=1))) / len(inlier_indices)

            #Surface Area Deviation and Normals Consistency Error - Port over

        else:
            print("No points within the distance threshold for metric calculation.")

        return metrics