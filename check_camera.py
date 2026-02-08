import cv2

def check_camera(source: int = 0) -> None:
    """
    Checks if the camera source is available and displays the feed.
    
    Args:
        source (int): The camera index to check. Defaults to 0.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open camera {source}")
        return

    print(f"Camera {source} opened successfully. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        cv2.imshow('Camera Test', frame)
        if cv2.waitKey(1) == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    check_camera()
