import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from typing import *
from numpy._typing import * 

def split_image_into_blocks(image: ArrayLike, block_size: Any) -> ArrayLike:
    '''
    Split an image into smaller blocks of the specified block size. 
    The function takes an input image and a block size, which can be either an integer or a tuple of two integers representing the height and width of the blocks. 

    Args:
        - image: input image as a numpy array.
        - block_size: the size of the blocks, either an integer or a tuple of two integers representing the height and width of the blocks.

    Returns:
        - blocks: a numpy array containing the image divided into smaller blocks of the specified size.
    '''
    if isinstance(block_size,int):
        block_height =  block_width = block_size
    else : 
        assert len(block_size) == 2 
        block_height, block_width = block_size
    height, width = image.shape[:2]
    
    n_blocks_height = height // block_height
    n_blocks_width = width // block_width
    blocks = np.zeros((n_blocks_height, n_blocks_width, block_height, block_width), dtype=image.dtype)
    for i in range(n_blocks_height):
        for j in range(n_blocks_width):
            block = image[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width]
            blocks[i, j] = block
    return blocks
    
def Calculate_SAD(block1: np.ndarray, block2: np.ndarray):
    """
    Calculate the Sum of Absolute Differences (SAD) between two blocks.

    Args:
        - block1 (ArrayLike): The first block of pixels.
        - block2 (ArrayLike): The second block of pixels.

    Returns:
        - SAD: The SAD value between the two blocks.
    """
    return np.sum(np.abs(block1 - block2))

def pad_zeros_around_image(image: np.ndarray, pad_size: int or Tuple[int, int]) -> ArrayLike:
    '''
    Pads zeros around an input image with a specified pad size.
    
    Args:
        - image (np.ndarray): The input image to be padded.
        - pad_size (int or Tuple[int, int]): The size of padding to be added. 
            If an integer is provided, pad_size is applied equally to both height and width of the image.
            If a tuple of two integers is provided, the first integer represents the padding to be added 
            to the top and bottom of the image, and the second integer represents the padding to be added 
            to the left and right of the image.
            
    Returns:
        - ArrayLike: A new image with zeros padded around it, with the size increased by 2 times pad_size. 
    '''
    
    if isinstance(pad_size,int):
        pad_size_height =  pad_size_width = pad_size
    else : 
        assert len(pad_size) == 2 
        pad_size_height, pad_size_width = pad_size
    
    new_image = np.zeros((image.shape[0] + 2*pad_size_height, image.shape[1] + 2*pad_size_width))
    
    new_image[pad_size_height : new_image.shape[0] - pad_size_height, pad_size_width : new_image.shape[1] - pad_size_width] = image
    
    return new_image.astype(np.uint8)

def calculate_sad_with_surrounding_blocks(ref_frame: np.ndarray, block: np.ndarray, start: Tuple[int, int], end = None, stride = 1) -> ArrayLike:
    '''
    This function calculates the Sum of Absolute Differences (SAD) between a block and the surrounding blocks within a frame. 
    It then returns the position of the surrounding block with the minimum SAD.

    Args:
    - ref_frame: a numpy array representing the reference frame.
    - block: a numpy array representing the block for which SAD will be calculated.
    - start: a tuple representing the starting position of the block.
    - end: a tuple representing the ending position of the block (default is None).
    - stride: an integer representing the stride (default is 1).
    Returns:
    - A tuple containing the position of the center of the block in the reference frame and the position of the center of the block in the current frame.
    '''
    assert len(start) == 2 
    block_height, block_width = block.shape
    start_height, start_width = start
    if end is None:
        end_height, end_width = start_height + block_height, start_width + block_width
    else: 
        assert len(end) == 2  
        end_height, end_width = end
    sad = []
    
    pad_size_height, pad_size_width = pad_size = block_height//2, block_width//2
    
    padded_img = pad_zeros_around_image(ref_frame, pad_size)
    
    for i in range(start_height + pad_size_height, end_height + pad_size_width, stride):
        for j in range(start_width + pad_size_width, end_width + pad_size_width, stride):
            sad.append(
                Calculate_SAD(
                    padded_img[i - block_height//2 : i + block_height//2 , j - block_width//2 : j + block_width//2],
                    block
                )
            )
    
    sad = np.array(sad).reshape(block.shape)
    min_sad = np.min(sad)
    idx_sad = np.where(
        min_sad == sad
    )
    
    center_block_in_current_frame = start_height + block_height//2, start_width + block_width//2
    center_block_in_reference_frame = start_height + idx_sad[0][0], start_width + idx_sad[1][0]
    return center_block_in_reference_frame, center_block_in_current_frame

def sliding_window_blocks(ref_frame: np.ndarray, blocks: np.ndarray) -> ArrayLike:
    '''
    Slides a window of blocks across the given reference frame and calculates the motion vectors for each block.

    Args:
        - ref_frame: a 2D numpy array representing the reference frame to analyze.
        - blocks: a 4D numpy array representing the blocks to use for motion estimation.
              The shape of the array is (n_blocks_height, n_blocks_width, block_height, block_width).

    Returns:
        - A list of dictionaries, where each dictionary represents a motion vector for a block.
            The dictionary contains the start and end points of the motion vector.
    '''
    n_blocks_height, n_blocks_width, block_height, block_width = blocks.shape
    
    motion_vectors = []
    
    for i in range(n_blocks_height):
        for j in range(n_blocks_width):
            center_block_in_reference_frame, center_block_in_current_frame = calculate_sad_with_surrounding_blocks(
                ref_frame, blocks[i,j], start = (i * block_height ,j * block_width)
            )

            vector = {
                # 'coordinates_block': (i,j),
                'start_point': center_block_in_reference_frame,
                'end_point': center_block_in_current_frame
            }
            
            motion_vectors.append(vector)

    return motion_vectors

def visualize_motion_vectors(frame: np.ndarray, motion_vectors: ArrayLike):
    vectors_map = np.zeros_like(frame)
    
    for vector in motion_vectors:
        start_point = vector['start_point']
        end_point = vector['end_point']
        cv.line(vectors_map, start_point, end_point, color = (255,255,255), thickness = 1)
    return vectors_map
        
def main(video_filename):
    block_size = 16
    img_size = (320,320)
    video = cv.VideoCapture(filename=video_filename)
    pre_frame = None
    while True:
        ret, frame = video.read()
        
        if ret:
            frame = cv.resize(frame, img_size)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if pre_frame is None:
                pre_frame = frame
                continue
            
            blocks = split_image_into_blocks(frame, block_size)
            vector = sliding_window_blocks(pre_frame, blocks)
            motion_vectors_map = visualize_motion_vectors(pre_frame, vector)
            cv.imshow('motion vectors', motion_vectors_map)     
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break 
        else:
            break
        
    video.release()
    cv.destroyAllWindows()
    
if __name__ == '__main__':
    filename = 'video_test.mp4'
    main(filename)