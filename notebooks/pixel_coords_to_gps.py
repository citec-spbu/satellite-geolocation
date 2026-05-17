def pixel_coords_to_jps(tl: tuple, br: tuple, pixel_coords: tuple, img_width: int=768, img_height: int=768) -> tuple:
    """
    Convert pixel coordinates to JPS coordinates.

    Args:
        tl (tuple): Top-left gps coordinates (lat, lon).
        br (tuple): Bottom-right gps coordinates (lat, lon).
        img_width (int): Width of the image.
        img_height (int): Height of the image.
        pixel_coords (tuple): Pixel coordinates (x, y).

    Returns:
        tuple: JPS coordinates (lat, lon).
    Примечание:
        Ha наших тестовых данных все фотографии квадратные 768x768 пикселей, 
        однако при использовании не на тестовом датасете также потребуются ширина и высота исходного изображения satellite.
    """
    lat_range = tl[0] - br[0]
    lon_range = br[1] - tl[1]

    lat_per_pixel = lat_range / img_height
    lon_per_pixel = lon_range / img_width

    jps_x = tl[1] + pixel_coords[0] * lon_per_pixel
    jps_y = tl[0] - pixel_coords[1] * lat_per_pixel

    return (jps_y, jps_x)


# if __name__ == "__main__":
#     tl = (120.33490057840369, 30.326315767763095)
#     br = (120.33946217810171, 30.322379961513835) 
#     pixel_coords = (399, 262)

#     jps_coords = pixel_coords_to_jps(tl, br, pixel_coords)
#     print(f"JPS coordinates: {jps_coords}")

#ans=          120.336456749134   30.324270993422658
#real=         120.33645688888889 30.324266833333333