import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np

class ICPRegistration:
    def __init__(self, fixed_landmarks, moving_landmarks, fixed_image):
        """
        Initializes the ICPRegistration class.

        Args:
            fixed_landmarks (list): List of 3D landmark points in the fixed/base (MRI) space.
            moving_landmarks (list): List of 3D landmark points in the moving (physical) space.
            fixed_image (sitk.Image): The fixed/base MRI image.
        """
        self.fixed_landmarks = fixed_landmarks
        self.moving_landmarks = moving_landmarks
        self.fixed_image = fixed_image

    def perform_icp_registration(self, outlier_rejection_threshold=10.0):
        """
        Performs ICP registration with outlier rejection.

        Args:
            outlier_rejection_threshold (float): Threshold distance for outlier rejection in millimeters.

        Returns:
            vtk.vtkTransform: The resulting transformation.
            list: List of inlier indices after outlier rejection.
        """
        # Convert landmark lists to vtkPoints
        fixed_points = vtk.vtkPoints()
        moving_points = vtk.vtkPoints()
        for point in self.fixed_landmarks:
            fixed_points.InsertNextPoint(point)
        for point in self.moving_landmarks:
            moving_points.InsertNextPoint(point)

        # Create polydata objects for the landmarks
        fixed_polydata = vtk.vtkPolyData()
        fixed_polydata.SetPoints(fixed_points)
        moving_polydata = vtk.vtkPolyData()
        moving_polydata.SetPoints(moving_points)

        # Initialize ICP transform
        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(moving_polydata)
        icp.SetTarget(fixed_polydata)
        icp.GetLandmarkTransform().SetModeToSimilarity()  
        icp.SetMaximumNumberOfIterations(100)
        icp.StartByMatchingCentroidsOn()
        icp.SetCheckMeanDistanceOn()
        icp.Update()

        # Get the transformation matrix
        transform = icp.GetLandmarkTransform()

        # Apply the transformation to the moving landmarks
        transformed_points = vtk.vtkPoints()
        transform.TransformPoints(moving_points, transformed_points)

        # Outlier rejection based on distance threshold
        inlier_indices = []
        for i in range(moving_points.GetNumberOfPoints()):
            original_point = moving_points.GetPoint(i)
            transformed_point = transformed_points.GetPoint(i)
            distance = np.sqrt(vtk.vtkMath.Distance2BetweenPoints(original_point, transformed_point))
            if distance <= outlier_rejection_threshold:
                inlier_indices.append(i)

        return transform, inlier_indices

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
