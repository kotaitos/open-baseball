from abc import ABC, abstractmethod
from typing import Any


class BaseAnalyzer(ABC):
    """
    すべての解析手法の基底クラス。
    """

    @abstractmethod
    def analyze_frame(self, frame, frame_idx: int, fps: float, context: dict = None) -> Any:
        """
        フレームごとに解析を実行します。
        """
        pass

    @abstractmethod
    def reset(self):
        """
        解析状態をリセットします。
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """
        解析手法の名前を返します。
        """
        pass
