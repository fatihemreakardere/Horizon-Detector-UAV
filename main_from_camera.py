import cv2
import numpy as np

class HorizonDetector:
    def __init__(self):
        self.slope_memory = []  # Memory for averaging slopes
        self.intercept_memory = []  # Memory for averaging intercepts
        self.memory_size = 15  # Size of the memory for averaging

    def preprocess_image(self, frame):
        """ Resize the image to a standard size to simplify processing """
        min_length = min(frame.shape[:2])
        if frame.shape[1] == min_length:
            frame = frame[(frame.shape[0] - min_length) // 2:frame.shape[0] - (frame.shape[0] - min_length) // 2, :]
        else:
            frame = frame[:, (frame.shape[1] - min_length) // 2:frame.shape[1] - (frame.shape[1] - min_length) // 2]
        return cv2.resize(frame, (100, 100), interpolation=cv2.INTER_LINEAR), frame

    def apply_thresholds(self, frame):
        """ Apply combined Otsu and adaptive thresholding to highlight features """
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gaussian_blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        _, otsu_mask = cv2.threshold(gaussian_blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adaptive_mask = cv2.adaptiveThreshold(gaussian_blurred_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                              cv2.THRESH_BINARY, 11, 2)
        combined_mask = cv2.addWeighted(otsu_mask, 0.5, adaptive_mask, 0.5, 1)
        _, final_mask = cv2.threshold(combined_mask, 128, 255, cv2.THRESH_BINARY)
        return final_mask, gaussian_blurred_frame
    
    def custom_mask_largest_white(self, mask):
        """ Identify the largest connected white region in the mask """
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=4)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        custom_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
        return custom_mask

    def custom_mask_largest_black(self, mask):
        """ Identify the largest connected black region in the mask """
        inverted_mask = cv2.bitwise_not(mask)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted_mask, connectivity=8)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        custom_mask = np.where(labels == largest_label, 0, 255).astype(np.uint8)
        return custom_mask
    
    def find_longest_contour(self, frame):
        """ Find the longest contour in the frame to potentially represent the horizon """
        edges = cv2.Canny(frame, 100, 200)
        dilated_edges = cv2.dilate(edges, None, iterations=1)
        contours, _ = cv2.findContours(dilated_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea) if contours else None

    def update_memory(self, slope, intercept):
        """ Update memory buffers for slope and intercept """
        if len(self.slope_memory) >= self.memory_size:
            self.slope_memory.pop(0)
            self.intercept_memory.pop(0)
        self.slope_memory.append(slope)
        self.intercept_memory.append(intercept)

    def get_average_line(self):
        """ Calculate the average slope and intercept from memory """
        avg_slope = np.mean(self.slope_memory) if self.slope_memory else 0
        avg_intercept = np.mean(self.intercept_memory) if self.intercept_memory else 0
        return avg_slope, avg_intercept
    
    def draw_contour_and_line(self, frame1, frame2, contour, original_frame):
        if contour is not None:
            cv2.drawContours(frame1, [contour], -1, (0, 0, 255), 2)
            original_frame = cv2.resize(original_frame, (frame1.shape[1]*2, frame1.shape[0]*2), interpolation=cv2.INTER_LINEAR)
            # Scale the contour for the original frame
            scale_x = original_frame.shape[1] / frame1.shape[1]
            scale_y = original_frame.shape[0] / frame1.shape[0]

            # Using RANSAC to fit line robustly
            [vx, vy, x, y] = cv2.fitLine(np.array(contour), cv2.DIST_L2, 0, 0.01, 0.01)
            _, cols = frame2.shape[:2]
            slope = vy / vx
            intercept = y - slope * x
            self.update_memory(slope, intercept)
            avg_slope, avg_intercept = self.get_average_line()
            lefty_avg = int(avg_intercept)
            righty_avg = int((avg_slope * cols) + avg_intercept)

            # Draw line on the processed frame
            cv2.line(frame2, (cols - 1, righty_avg), (0, lefty_avg), (0, 255, 0), 2)

            # Draw line on the original frame
            cols_original = original_frame.shape[1]
            lefty_original = int(avg_intercept * scale_y)
            righty_original = int((avg_slope * cols_original) + avg_intercept * scale_y)
            cv2.line(original_frame, (cols_original - 1, righty_original), (0, lefty_original), 2)

        return frame1, frame2, original_frame

def process_video():
    cap = cv2.VideoCapture(0)  # Use the default camera
    detector = HorizonDetector()
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame, original_frame = detector.preprocess_image(frame)
        processed_frame2, original_frame = detector.preprocess_image(frame)
        thresholded_frame, gausian_blur = detector.apply_thresholds(processed_frame)
        
        largest_white_mask = detector.custom_mask_largest_white(thresholded_frame)
        largest_black_mask = detector.custom_mask_largest_black(largest_white_mask)

        longest_contour = detector.find_longest_contour(largest_black_mask)
        frame_with_contour, frame_with_line, original_frame = detector.draw_contour_and_line(processed_frame, processed_frame2, longest_contour, original_frame)

        cv2.imshow('Processed Frame', processed_frame)
        cv2.imshow('Thresholded Frame', thresholded_frame)
        cv2.imshow('Largest White Component', largest_white_mask)
        cv2.imshow('Largest Black Component', largest_black_mask)
        cv2.imshow('Horizon Detection - Contour', frame_with_contour)
        cv2.imshow('Horizon Detection - Line', frame_with_line)
        cv2.imshow("Gaussian Blurred Frame", gausian_blur)
        cv2.imshow("Original Frame", original_frame)
        

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
