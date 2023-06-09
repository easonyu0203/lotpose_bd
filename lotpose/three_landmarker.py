from typing import Optional

import numpy as np

from lotpose.dtos.landmark_3d_dto import Landmark3dDto
from lotpose.dtos.mono_result_dto import MonoResultDto


class ThreeLandmarker:
    """get 3d landmark"""

    def process(self, mono_results: dict[int, MonoResultDto]) -> Optional[Landmark3dDto]:
        """process mono results and return 3d landmark"""

        target = mono_results[0]
        mono_landmark = target.result.pose_landmarks
        if mono_landmark is None or mono_landmark == []:
            return None
        mono_landmark = mono_landmark[0]

        landmark3d_value = np.array([[p.x * 5, -p.y * 5, p.z * 5, p.presence] for p in mono_landmark])
        landmark3d_value[:, :3] = landmark3d_value[:, :3] - landmark3d_value[0, :3] + np.array([0, 5, 0])

        landmark3d = Landmark3dDto(
            device_index=target.device_index,
            timestamp=target.timestamp,
            value=landmark3d_value
        )

        return landmark3d
