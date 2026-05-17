from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
from PIL.Image import Image as PILImage


class GalleryRepository(ABC):
    """Абстрактный интерфейс для хранилища изображений галереи и их эмбеддингов."""

    @abstractmethod
    def search_similar(
        self, embedding: np.ndarray, top_k: int
    ) -> List[Tuple[str, float]]:
        """
        Поиск top_k ближайших изображений по эмбеддингу.

        Args:
            embedding: нормализованный вектор эмбеддинга размерностью (512,)
            top_k: количество возвращаемых результатов

        Returns:
            Список кортежей (image_id, cosine_similarity), отсортированный по убыванию сходства
        """
        ...

    @abstractmethod
    def get_image(self, image_id: str) -> PILImage:
        """
        Получить PIL.Image по идентификатору.

        Args:
            image_id: уникальный идентификатор изображения

        Returns:
            PIL.Image объект

        Raises:
            KeyError: если изображение не найдено
        """
        ...

    @abstractmethod
    def add_image(
        self, image_bytes: bytes, embedding: np.ndarray, metadata: dict
    ) -> str:
        """
        Сохранить изображение и его эмбеддинг в хранилище.

        Args:
            image_bytes: байты изображения (JPEG/PNG)
            embedding: нормализованный вектор эмбеддинга
            metadata: словарь с метаданными (filename, location, content_type и т.д.)

        Returns:
            Уникальный идентификатор сохранённого изображения
        """
        ...

    @abstractmethod
    def count(self) -> int:
        """
        Количество записей в галерее.

        Returns:
            Число сохранённых изображений
        """
        ...

    @abstractmethod
    def delete_image(self, image_id: str) -> bool:
        """
        Удалить изображение и его эмбеддинг из хранилища.

        Args:
            image_id: идентификатор изображения

        Returns:
            True если удаление успешно, False если изображение не найдено
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """Полностью очистить галерею (удалить все изображения и эмбеддинги)."""
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """
        Проверка доступности хранилища.

        Returns:
            True если хранилище доступно и работает корректно
        """
        ...
