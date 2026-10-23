#!/usr/bin/env python
#
# Copyright 2008 Willow Garage, Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the Willow Garage, Inc. nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import copy
import unittest

from geometry_msgs.msg import Transform, TransformStamped
import numpy as np
from sensor_msgs.msg import PointField
from sensor_msgs_py.point_cloud2 import (
    create_cloud, create_cloud_xyz32, read_points, read_points_numpy)
from std_msgs.msg import Header
import tf2_ros
from tf2_sensor_msgs import do_transform_cloud, transform_points


class PointCloudConversions(unittest.TestCase):

    def test_transform_extra_fields_owns_output(self):
        points = np.array([(1., 2., 3., 42), (4., 5., 6., 255)], dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('intensity', 'u4')])
        fields = [PointField(name=name, offset=i * 4, count=1, datatype=datatype)
                  for i, (name, datatype) in enumerate(
                      [('x', 7), ('y', 7), ('z', 7), ('intensity', 6)])]
        cloud = create_cloud(Header(frame_id='original'), fields, points)
        original = bytes(cloud.data)
        transform = TransformStamped()
        transform.header.frame_id = 'transformed'
        transform.transform.rotation.w = 1.0
        transform.transform.translation.x = 10.0
        output = do_transform_cloud(cloud, transform)
        actual = read_points(output)
        np.testing.assert_array_equal(actual['x'], points['x'] + 10)
        for field in ('y', 'z', 'intensity'):
            np.testing.assert_array_equal(actual[field], points[field])
        self.assertEqual(output.header.frame_id, 'transformed')
        self.assertEqual(cloud.header.frame_id, 'original')
        self.assertEqual(bytes(cloud.data), original)
        self.assertFalse(np.shares_memory(np.asarray(output.data), np.asarray(cloud.data)))
        actual['x'] = 0
        self.assertEqual(bytes(cloud.data), original)

    def setUp(self):
        self.points = np.array([[1, 2, 0], [10, 20, 30]], dtype=np.float32)

        self.point_cloud_in = create_cloud_xyz32(
            Header(frame_id='test'),
            self.points)

    def test_transform_padded_points(self):
        # PCL point types pad points, PointXYZI to 32 bytes.
        fields = [PointField(name=name, offset=offset, count=1, datatype=PointField.FLOAT32)
                  for name, offset in [('x', 0), ('y', 4), ('z', 8), ('intensity', 16)]]
        points = np.zeros(3, dtype={
            'names': ['x', 'y', 'z', 'intensity'], 'formats': ['f4'] * 4,
            'offsets': [0, 4, 8, 16], 'itemsize': 32})
        points['x'] = [1., 2., 3.]
        points['intensity'] = [7., 8., 9.]
        cloud = create_cloud(Header(frame_id='original'), fields, points, point_step=32)
        transform = TransformStamped()
        transform.transform.rotation.w = 1.0
        transform.transform.translation.y = 5.0
        output = do_transform_cloud(cloud, transform)
        actual = read_points(output)
        np.testing.assert_array_equal(actual['x'], [1., 2., 3.])
        np.testing.assert_array_equal(actual['y'], [5., 5., 5.])
        np.testing.assert_array_equal(actual['intensity'], [7., 8., 9.])
        self.assertEqual(32, output.point_step)

    def test_simple_transform(self):
        k = 300.0
        transform_translate_xyz = TransformStamped()
        transform_translate_xyz.transform.translation.x = k
        transform_translate_xyz.transform.translation.y = k
        transform_translate_xyz.transform.translation.z = k
        # no rotation so we only set w
        transform_translate_xyz.transform.rotation.w = 1.0

        # Copy current state of the original message for later check
        old_data = copy.deepcopy(self.point_cloud_in.data)

        # Apply transform
        point_cloud_transformed = do_transform_cloud(
            self.point_cloud_in, transform_translate_xyz)

        # Expected output
        expected_coordinates = self.points + k

        new_points = read_points_numpy(point_cloud_transformed)
        self.assertTrue(np.allclose(expected_coordinates, new_points))

        # checking no modification in input cloud
        self.assertEqual(old_data, self.point_cloud_in.data)

    def test_simple_rotation_transform(self):
        # Create a transform containing only a 180° rotation around the x-axis
        transform_rot = TransformStamped()
        transform_rot.transform.rotation.x = 1.0
        transform_rot.transform.rotation.w = 0.0

        # Apply transform
        point_cloud_transformed = do_transform_cloud(self.point_cloud_in, transform_rot)

        # Expected output
        expected_coordinates = np.array([[1, -2, 0], [10, -20, -30]], dtype=np.float32)

        # Compare to ground truth
        new_points = read_points_numpy(point_cloud_transformed)
        self.assertTrue(np.allclose(expected_coordinates, new_points))

    def test_rotation_and_translation_transform(self):
        # Create a transform combining a 100m z translation with
        # a 180° rotation around the x-axis
        transform = TransformStamped()
        transform.transform.translation.z = 100.0
        transform.transform.rotation.x = 1.0
        transform.transform.rotation.w = 0.0

        # Apply transform
        point_cloud_transformed = do_transform_cloud(
            self.point_cloud_in, transform)

        # Expected output
        expected_coordinates = np.array([[1, -2, 100], [10, -20, 70]], dtype=np.float32)

        # Compare to ground truth
        new_points = read_points_numpy(point_cloud_transformed)
        self.assertTrue(np.allclose(expected_coordinates, new_points))

    def test_direct_transform(self):
        # Create a transform combining a 100m z translation with
        # a 180° rotation around the x-axis
        transform = Transform()
        transform.translation.z = 100.0
        transform.rotation.x = 1.0
        transform.rotation.w = 0.0

        # Transform points
        points = transform_points(self.points, transform)

        # Expected output
        expected_coordinates = np.array([[1, -2, 100], [10, -20, 70]], dtype=np.float32)

        # Compare to ground truth
        self.assertTrue(np.allclose(expected_coordinates, points))

    def test_tf2_ros_transform(self):
        # Our target frame
        target_frame_name = 'base_footprint'

        # We need to create a local test tf buffer
        tf_buffer = tf2_ros.Buffer()

        # We need to fill this tf_buffer with a possible transform
        # So we create a transform with a 100m z translation
        transform = TransformStamped()
        transform.header.frame_id = 'test'
        transform.child_frame_id = target_frame_name
        transform.transform.translation.z = 100.0
        transform.transform.rotation.w = 1.0

        # Set the new transform in our local tf_buffer
        tf_buffer.set_transform_static(transform, 'unittest')

        point_cloud_transformed = tf_buffer.transform(self.point_cloud_in, target_frame_name)

        # Check if our pointloud is in the correct frame
        self.assertEqual(point_cloud_transformed.header.frame_id, target_frame_name)

        # Check if the points are viewed from the target frame (inverse of the transform above)
        self.assertTrue(np.allclose(
            read_points_numpy(point_cloud_transformed),
            self.points -
            np.array([
                transform.transform.translation.x,
                transform.transform.translation.y,
                transform.transform.translation.z])))

    def test_non_coordinate_fields(self):
        # Test if point clouds with additional fields (non xyz) behave as desired

        # Create points with an additional field in addition to the x, y, and z ones
        points = np.array([[1, 2, 0, 9], [10, 20, 30, 9]], dtype=np.float32)

        # Create the field layout
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='a', offset=12, datatype=PointField.FLOAT32, count=1)]

        # Create cloud with four fields
        point_cloud = create_cloud(
            Header(frame_id='test'),
            fields,
            points)

        # Create the test transformation
        k = 300.0
        transform_translate_xyz = TransformStamped()
        transform_translate_xyz.transform.translation.x = k
        transform_translate_xyz.transform.translation.y = k
        transform_translate_xyz.transform.translation.z = k
        # no rotation so we only set w
        transform_translate_xyz.transform.rotation.w = 1.0

        # Copy current state of the original message for later check
        old_data = copy.deepcopy(point_cloud.data)

        # Check if the created point cloud contains our points
        self.assertTrue(np.allclose(
            read_points_numpy(point_cloud),
            points))

        # Apply transform
        point_cloud_transformed = do_transform_cloud(
            point_cloud, transform_translate_xyz)

        # Expected output, the last field should be unaltered
        expected_coordinates = points + np.array([k, k, k, 0])

        # Check if this is the case
        new_points = read_points_numpy(point_cloud_transformed)
        self.assertTrue(np.allclose(expected_coordinates, new_points))

        # Checking for no modification in input cloud
        self.assertEqual(old_data, point_cloud.data)


if __name__ == '__main__':
    unittest.main()
