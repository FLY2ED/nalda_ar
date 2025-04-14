import numpy as np
import cv2 as cv
from datetime import datetime

def load_camera_params(calibration_file='calibration_results/calibration_data.npz'):
    """사전에 캘리브레이션한 카메라 파라미터를 로드하는 함수"""
    try:
        data = np.load(calibration_file)
        camera_matrix = data['camera_matrix']
        dist_coeffs = data['distortion_coeffs']
        return camera_matrix, dist_coeffs
    except:
        print(f"캘리브레이션 파일을 찾을 수 없음: {calibration_file}")
        return None, None

def estimate_pose(image, camera_matrix, dist_coeffs, board_pattern=(10, 7), board_cellsize=0.025):
    """체스보드 패턴으로 카메라 포즈를 계산하는 함수"""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # 체스보드 패턴에 맞게 좌표 생성
    objp = np.zeros((board_pattern[0] * board_pattern[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_pattern[0], 0:board_pattern[1]].T.reshape(-1, 2) * board_cellsize
    
    # 체스보드 코너 찾기
    ret, corners = cv.findChessboardCorners(gray, board_pattern, None)
    
    if ret:
        # 코너 위치 정밀하게 조정
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 카메라 포즈 계산
        _, rvec, tvec = cv.solvePnP(objp, corners, camera_matrix, dist_coeffs)
        
        return ret, corners, rvec, tvec
    
    return ret, None, None, None

def draw_augmented_image(frame, image_to_augment, rvec, tvec, camera_matrix, dist_coeffs, board_pattern=(10, 7), board_cellsize=0.025, corners=None):
    """체스보드 위에 이미지와 텍스트를 투영하는 함수"""
    # 원본 이미지 비율 계산
    h, w = image_to_augment.shape[:2]

    # 체스보드 크기 설정
    board_width = board_pattern[0]
    board_height = board_pattern[1]
    
    # 이미지 위치 설정 
    grid_width = 4 
    grid_height = 4  
    grid_start_x = 1  # 왼쪽 여백
    grid_start_y = 1  # 위쪽 여백
    
    # 'st' 텍스트 위치 설정
    text_grid_width = 2 
    text_grid_height = 2
    text_grid_start_x = grid_start_x + grid_width + 1  # 이미지 옆에 1칸 띄우기
    text_grid_start_y = grid_start_y + 1
    
    result = frame.copy()
    
    # 체스보드 코너 정보로 좌표 계산
    if corners is not None:
        corners = corners.reshape(board_height, board_width, 2)
        
        # 이미지 영역 좌표 계산
        tl_idx = (grid_start_y, grid_start_x)
        tr_idx = (grid_start_y, grid_start_x + grid_width)
        br_idx = (grid_start_y + grid_height, grid_start_x + grid_width)
        bl_idx = (grid_start_y + grid_height, grid_start_x)
        
        # 텍스트 영역 좌표 계산
        text_tl_idx = (text_grid_start_y, text_grid_start_x)
        text_tr_idx = (text_grid_start_y, text_grid_start_x + text_grid_width)
        text_br_idx = (text_grid_start_y + text_grid_height, text_grid_start_x + text_grid_width)
        text_bl_idx = (text_grid_start_y + text_grid_height, text_grid_start_x)
        
        # 좌표가 유효한지 확인
        if (tr_idx[1] < board_width and br_idx[0] < board_height and
            text_tr_idx[1] < board_width and text_br_idx[0] < board_height):
            
            # 이미지 영역 코너 좌표
            img_points = np.float32([
                corners[tl_idx],
                corners[tr_idx],
                corners[br_idx],
                corners[bl_idx]
            ])
            
            # 텍스트 영역 코너 좌표
            text_points = np.float32([
                corners[text_tl_idx],
                corners[text_tr_idx],
                corners[text_br_idx],
                corners[text_bl_idx]
            ])
            
            # 원본 이미지 좌표
            pts_src = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
            
            # 이미지 호모그래피 계산
            H_img, _ = cv.findHomography(pts_src, img_points)
            
            # 이미지 투영
            warped_img = cv.warpPerspective(image_to_augment, H_img, (frame.shape[1], frame.shape[0]))
            
            # 이미지 마스크 생성
            img_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv.fillConvexPoly(img_mask, img_points.astype(int), 255)
            img_mask = cv.GaussianBlur(img_mask, (3, 3), 0.5)  # 경계 부드럽게
            img_mask_3channel = cv.cvtColor(img_mask, cv.COLOR_GRAY2BGR) / 255.0
            
            # 이미지 합성
            result = (1.0 - img_mask_3channel) * frame + img_mask_3channel * warped_img
            
            # 'st' 텍스트 이미지 생성 - 이미지
            text_img_size = 200 
            text_img = np.ones((text_img_size, text_img_size, 3), dtype=np.uint8) * 255
            
            # 텍스트 스타일 설정
            font_scale = 5.0
            thickness = 10
            font = cv.FONT_HERSHEY_SIMPLEX
            text = "st"
            
            # 텍스트 중앙 배치
            (text_width, text_height), _ = cv.getTextSize(text, font, font_scale, thickness)
            text_x = (text_img_size - text_width) // 2
            text_y = (text_img_size + text_height) // 2
            
            # 텍스트 그리기
            cv.putText(text_img, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv.LINE_AA)
            
            # 텍스트 이미지 좌표
            text_img_pts = np.float32([[0, 0], [text_img_size-1, 0], 
                                      [text_img_size-1, text_img_size-1], 
                                      [0, text_img_size-1]])
            
            # 텍스트 호모그래피 계산
            H_text, _ = cv.findHomography(text_img_pts, text_points)
            
            # 텍스트 이미지 투영
            warped_text = cv.warpPerspective(text_img, H_text, (frame.shape[1], frame.shape[0]))
            
            # 텍스트 마스크 생성
            text_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv.fillConvexPoly(text_mask, text_points.astype(int), 255)
            text_mask = cv.GaussianBlur(text_mask, (3, 3), 0.5)
            text_mask_3channel = cv.cvtColor(text_mask, cv.COLOR_GRAY2BGR) / 255.0
            
            # 텍스트 합성
            result = (1.0 - text_mask_3channel) * result + text_mask_3channel * warped_text
            
            return result.astype(np.uint8)
    
    # 코너 검출 실패하면 원본 프레임 반환
    return frame

def main():
    # 카메라 캘리브레이션 데이터 불러오기
    camera_matrix, dist_coeffs = load_camera_params()
    if camera_matrix is None or dist_coeffs is None:
        print("캘리브레이션 파일을 찾을 수 없음.")
        return
    
    # AR에 사용할 이미지 로드
    augment_image = cv.imread('image.png')
    if augment_image is None:
        print("투영할 이미지 파일을 찾을 수 없음.")
        return
    
    # 이미지 크기 최적화
    max_img_size = 800
    h, w = augment_image.shape[:2]
    if max(h, w) > max_img_size:
        scale = max_img_size / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        augment_image = cv.resize(augment_image, new_size, interpolation=cv.INTER_AREA)
    
    # 카메라 설정
    cap = cv.VideoCapture(0)
    
    # 체스보드 설정
    board_pattern = (10, 7)
    board_cellsize = 0.025  # 셀 크기 (m)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 영상을 가져올 수 없음.")
            break
        
        # 화면 크기 조정
        frame = cv.resize(frame, (800, 600))
        
        # 체스보드 인식 및 포즈 계산
        ret, corners, rvec, tvec = estimate_pose(frame, camera_matrix, dist_coeffs, board_pattern, board_cellsize)
        
        if ret:
            # AR 구현
            frame = draw_augmented_image(frame, augment_image, rvec, tvec, camera_matrix, dist_coeffs, board_pattern, board_cellsize, corners)
            
            # 체스보드 코너 표시
            cv.drawChessboardCorners(frame, board_pattern, corners, ret)
        
        # 결과 화면 표시
        cv.imshow('Augmented Reality', frame)
        
        # 'q' 키로 종료
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main() 