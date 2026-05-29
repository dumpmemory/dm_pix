# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for pad_to_size with channels-first (CHW) layout."""

from absl.testing import absltest
from dm_pix._src import augment
import jax.numpy as jnp
import numpy as np


class PadToSizeChannelsFirstTest(absltest.TestCase):
  """Regression tests for pad_to_size with channel_axis != -1."""

  def test_pad_to_size_hwc_shape(self):
    """Channels-last (HWC) baseline: output shape must be (H', W', C)."""
    image = jnp.zeros((4, 4, 3))  # HWC
    result = augment.pad_to_size(image, target_height=6, target_width=6,
                                 channel_axis=-1)
    self.assertEqual(result.shape, (6, 6, 3))

  def test_pad_to_size_chw_shape(self):
    """Channels-first (CHW) with channel_axis=0: output shape must be (C, H', W').

    Bug: pad_to_size hardcodes pad_width as ((top,bot), (left,right), (0,0)),
    which treats the channel axis as the last dimension regardless of
    channel_axis. For CHW input (C, H, W), this pads along C and H but not W,
    producing a wrong shape.
    """
    image = jnp.zeros((3, 4, 4))  # CHW: C=3, H=4, W=4
    result = augment.pad_to_size(image, target_height=6, target_width=6,
                                 channel_axis=0)
    self.assertEqual(result.shape, (3, 6, 6))

  def test_pad_to_size_chw_values(self):
    """CHW padding must center the original image with zeros around it."""
    # 1x1 single-pixel image, CHW with 1 channel
    image = jnp.ones((1, 1, 1))  # C=1, H=1, W=1
    result = augment.pad_to_size(image, target_height=3, target_width=3,
                                 channel_axis=0)
    self.assertEqual(result.shape, (1, 3, 3))
    # The center pixel should be 1; surrounding pixels should be 0.
    np.testing.assert_array_equal(result[0, 1, 1], 1.0)
    np.testing.assert_array_equal(result[0, 0, :], 0.0)
    np.testing.assert_array_equal(result[0, 2, :], 0.0)
    np.testing.assert_array_equal(result[0, :, 0], 0.0)
    np.testing.assert_array_equal(result[0, :, 2], 0.0)

  def test_pad_to_size_batch_chw_shape(self):
    """Batched CHW (B, C, H, W): output shape must be (B, C, H', W')."""
    image = jnp.zeros((2, 3, 4, 4))  # B=2, C=3, H=4, W=4
    result = augment.pad_to_size(image, target_height=6, target_width=6,
                                 channel_axis=1)
    self.assertEqual(result.shape, (2, 3, 6, 6))


if __name__ == '__main__':
  absltest.main()
