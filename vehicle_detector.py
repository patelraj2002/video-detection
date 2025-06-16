import cv2
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import os
from datetime import datetime
import glob

class EnhancedMultiVideoVehicleDetectionEngine:
    def __init__(self, model_name='yolov8n.pt', data_folder='data'):        
        self.data_folder = data_folder
        
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
        except ImportError:
            os.system("pip install ultralytics")
            from ultralytics import YOLO
            self.model = YOLO(model_name)        
        self.vehicle_classes = {
            2: 'car',
            3: 'motorcycle', 
            5: 'bus',
            7: 'truck'
        }
        
        self.vehicle_type_mapping = {
            'car': ['sedan', 'coupe', 'hatchback', 'suv'],
            'truck': ['pickup', 'delivery', 'semi'],
            'bus': ['city_bus', 'mini_bus', 'coach'],
            'motorcycle': ['motorcycle', 'scooter']
        }
        
        self.frame_results = []
        self.class_counts = defaultdict(int)
        self.video_info = {}
        
    def discover_videos(self):
        if not os.path.exists(self.data_folder):
            print(f"Data folder '{self.data_folder}' not found!")
            return []
        
        video_patterns = [
            os.path.join(self.data_folder, "*.mp4"),
            os.path.join(self.data_folder, "*.MP4"),
        ]
        
        all_videos = []
        for pattern in video_patterns:
            all_videos.extend(glob.glob(pattern))
        
        def natural_sort_key(path):
            filename = os.path.basename(path)
            try:
                number = int(os.path.splitext(filename)[0])
                return (0, number)  # Priority to numbered files
            except:
                return (1, filename.lower())  # Other files come after
        
        all_videos.sort(key=natural_sort_key)
        return all_videos
    
    def display_available_videos(self, videos):
        """Display available videos with details"""
        print("\n" + "="*70)
        print("AVAILABLE VIDEOS FOR VEHICLE DETECTION")
        print("="*70)
        
        for i, video_path in enumerate(videos, 1):
            filename = os.path.basename(video_path)
            file_size = os.path.getsize(video_path) / (1024*1024)  # MB
            
            # Get video duration
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                print(f"   {i}. {filename} - {file_size:.1f}MB - {duration:.1f}s")
            else:
                print(f"   {i}. {filename} - {file_size:.1f}MB - [Cannot read duration]")
        
        print("="*70)
    
    def select_video(self):
        """Let user select which video to process"""
        videos = self.discover_videos()
        
        if not videos:
            print("No .mp4 videos found in the data folder!")
            print(f"Please ensure videos are placed in: {os.path.abspath(self.data_folder)}")
            return None
        
        if len(videos) == 1:
            print(f"Found 1 video: {os.path.basename(videos[0])}")
            return videos[0]
        
        self.display_available_videos(videos)
        
        while True:
            try:
                choice = input(f"Select video to test (1-{len(videos)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    print("Exiting...")
                    return None
                
                choice = int(choice)
                if 1 <= choice <= len(videos):
                    selected_video = videos[choice-1]
                    print(f"Selected: {os.path.basename(selected_video)}")
                    return selected_video
                else:
                    print(f"Please enter a number between 1 and {len(videos)}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
            except KeyboardInterrupt:
                print("\nProcess cancelled by user")
                return None
    
    def validate_video_for_vehicles(self, video_path, sample_frames=15):
        """Enhanced validation to check if video contains vehicles suitable for testing"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return False, "Cannot open video file"
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        vehicle_detections = 0
        frames_checked = 0
        vehicle_types_found = set()
        
        # Sample frames more thoroughly
        frame_indices = np.linspace(0, total_frames-1, min(sample_frames, total_frames), dtype=int)
        
        print(f"Validating video content by checking {len(frame_indices)} sample frames...")
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            frames_checked += 1
            
            # Run detection on sample frame
            results = self.model(frame, verbose=False)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        
                        if class_id in self.vehicle_classes and confidence > 0.3:
                            vehicle_detections += 1
                            vehicle_types_found.add(self.vehicle_classes[class_id])
        
        cap.release()
        
        vehicle_frame_ratio = vehicle_detections / max(frames_checked, 1)
        
        is_suitable = (
            vehicle_detections >= 5 and  
            vehicle_frame_ratio > 0.15 and 
            len(vehicle_types_found) >= 1  
        )
        
        validation_msg = (f"Found {vehicle_detections} vehicle detections in {frames_checked} frames. "
                         f"Vehicle types: {', '.join(vehicle_types_found) if vehicle_types_found else 'None'}")
        
        return is_suitable, validation_msg
    
    def classify_vehicle_subtype(self, bbox, class_name, frame_shape):
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 1
        area = width * height
        frame_area = frame_shape[0] * frame_shape[1]
        relative_size = area / frame_area
        
        if class_name == 'car':
            if relative_size > 0.02:  # Large cars
                if aspect_ratio > 2.5:
                    return 'sedan'
                elif aspect_ratio > 1.9:
                    return 'suv'
                else:
                    return 'coupe'
            elif aspect_ratio > 2.2:
                return 'sedan'
            elif aspect_ratio > 1.7:
                return 'hatchback'
            else:
                return 'coupe'
                
        elif class_name == 'truck':
            if relative_size > 0.035:  # Very large trucks
                return 'semi'
            elif aspect_ratio > 2.2:
                return 'pickup'
            else:
                return 'delivery'
                
        elif class_name == 'bus':
            if relative_size > 0.05:  # Very large bus
                return 'city_bus'
            elif aspect_ratio > 3.0:  # Long bus
                return 'coach'
            else:
                return 'mini_bus'
        elif class_name == 'motorcycle':
            if relative_size < 0.008:
                return 'scooter'
            else:
                return 'motorcycle'
        else:
            return class_name
    
    def process_video(self, video_path, output_dir='detection_results'):
        """Main video processing function"""
        self.frame_results = []
        self.class_counts = defaultdict(int)
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = f"{output_dir}_{video_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n🔍 Validating video content...")
        is_suitable, validation_msg = self.validate_video_for_vehicles(video_path)
        
        if not is_suitable:
            print("\n" + "="*60)
            print("VIDEO VALIDATION FAILED")
            print("="*60)
            print("This video is not suitable for the test.")
            print(f"Reason: {validation_msg}")
            print("Please use a video containing road traffic with vehicles.")
            print("Ensure the video shows cars, buses, trucks, or motorcycles.")
            print("="*60)
            return None
        
        print(f"Video validation passed: {validation_msg}")
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.video_info = {
            'filename': os.path.basename(video_path),
            'duration': duration,
            'total_frames': total_frames,
            'fps': fps,
            'width': width,
            'height': height
        }
        
        print(f"\nVIDEO INFORMATION:")
        print(f"  File: {self.video_info['filename']}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps:.1f}")
        print(f"  Resolution: {width}x{height}")
        print(f"  Processing every 5th frame...")
        
        # Setup video writer for compiled output
        output_video_path = f"{output_dir}/compiled_output_video.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps/5, (width, height))  
        frame_count = 0
        processed_frames = 0
        annotated_frames = []
        
        print(f"\nStarting detection process...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % 5 == 0:
                timestamp = frame_count / fps
                detections = self.detect_objects(frame, frame_count, timestamp)
                
                annotated_frame = self.draw_detections(frame.copy(), detections)
                annotated_frames.append(annotated_frame)
                
                out.write(annotated_frame)
                
                processed_frames += 1
                if processed_frames % 20 == 0:
                    print(f"   📸 Processed {processed_frames} frames...")
            
            frame_count += 1
        
        cap.release()
        out.release()
        
        print(f"Detection complete! Processed {processed_frames} frames total.")
        print(f"Compiled output video saved: {output_video_path}")
        
        self.generate_comprehensive_summary(output_dir)
        
        return self.frame_results
    
    def detect_objects(self, frame, frame_num, timestamp):
        results = self.model(frame, verbose=False)
        frame_data = {
            'frame_number': frame_num,
            'timestamp': timestamp,
            'objects': [],
            'class_diversity': 0,
            'total_objects': 0
        }
        
        classes_in_frame = set()
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    
                    if class_id in self.vehicle_classes and confidence > 0.25:
                        class_name = self.vehicle_classes[class_id]
                        bbox = box.xyxy[0].cpu().numpy()
                        
                        subtype = self.classify_vehicle_subtype(
                            bbox, class_name, frame.shape
                        )
                        
                        detection = {
                            'class': class_name,
                            'subtype': subtype,
                            'confidence': confidence,
                            'bbox': {
                                'x1': float(bbox[0]),
                                'y1': float(bbox[1]),
                                'x2': float(bbox[2]),
                                'y2': float(bbox[3])
                            }
                        }
                        
                        frame_data['objects'].append(detection)
                        classes_in_frame.add(subtype)
                        self.class_counts[subtype] += 1
        
        frame_data['class_diversity'] = len(classes_in_frame)
        frame_data['total_objects'] = len(frame_data['objects'])
        frame_data['unique_classes'] = list(classes_in_frame)
        
        self.frame_results.append(frame_data)
        return frame_data
    
    def draw_detections(self, frame, detections):
        colors = {
            'sedan': (0, 255, 0),      # Green
            'coupe': (0, 255, 255),    # Yellow
            'suv': (255, 0, 0),        # Blue
            'hatchback': (255, 255, 0), # Cyan
            'pickup': (0, 0, 255),     # Red
            'delivery': (255, 0, 255), # Magenta
            'semi': (128, 0, 128),     # Purple
            'city_bus': (0, 128, 255), # Orange
            'mini_bus': (255, 128, 0), # Light Blue
            'coach': (128, 255, 0),    # Lime
            'motorcycle': (255, 255, 255), # White
            'scooter': (128, 128, 128) # Gray
        }
        
        frame_info = f"Frame: {detections['frame_number']} | Time: {detections['timestamp']:.2f}s | Objects: {detections['total_objects']} | Diversity: {detections['class_diversity']}"
        cv2.rectangle(frame, (10, 10), (len(frame_info) * 8 + 20, 40), (0, 0, 0), -1)
        cv2.putText(frame, frame_info, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        for obj in detections['objects']:
            bbox = obj['bbox']
            class_name = obj['class']
            subtype = obj['subtype']
            confidence = obj['confidence']
            
            color = colors.get(subtype, (0, 255, 0))
            
            cv2.rectangle(frame, 
                         (int(bbox['x1']), int(bbox['y1'])),
                         (int(bbox['x2']), int(bbox['y2'])),
                         color, 2)
            
            label = f"{subtype.replace('_', ' ').title()} ({confidence:.2f})"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            cv2.rectangle(frame,
                         (int(bbox['x1']), int(bbox['y1']) - label_size[1] - 10),
                         (int(bbox['x1']) + label_size[0] + 5, int(bbox['y1'])),
                         color, -1)
            
            cv2.putText(frame, label,
                       (int(bbox['x1']) + 3, int(bbox['y1']) - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def generate_comprehensive_summary(self, output_dir):
        total_detections = sum(self.class_counts.values())
        total_frames = len(self.frame_results)
        
        max_diversity_frame = self.get_max_diversity_frame()
        
        json_output = {
            'video_info': self.video_info,
            'summary': {
                'total_frames_processed': total_frames,
                'total_detections': total_detections,
                'average_detections_per_frame': total_detections / max(total_frames, 1),
                'class_counts': dict(self.class_counts),
                'max_diversity_frame': max_diversity_frame,
                'processing_timestamp': datetime.now().isoformat(),
                'unique_vehicle_types': len(self.class_counts)
            },
            'frame_results': self.frame_results
        }
        
        with open(f"{output_dir}/detection_results.json", 'w') as f:
            json.dump(json_output, f, indent=2)
        
        self.create_enhanced_visualizations(output_dir)
        
        self.print_comprehensive_summary()
    
    def get_max_diversity_frame(self):
        if not self.frame_results:
            return None
        
        max_frame = max(self.frame_results, key=lambda x: x['class_diversity'])
        return {
            'frame_number': max_frame['frame_number'],
            'timestamp': max_frame['timestamp'],
            'class_diversity': max_frame['class_diversity'],
            'unique_classes': max_frame['unique_classes'],
            'total_objects': max_frame['total_objects']
        }
    
    def create_enhanced_visualizations(self, output_dir):
        if not self.class_counts:
            print("⚠️ No vehicles detected - skipping visualization generation")
            return
        
        plt.figure(figsize=(15, 8))
        
        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', 
                 '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
                 '#BB8FCE', '#85C1E9', '#F8C471', '#82E0AA']
        
        bars = plt.bar(classes, counts, color=colors[:len(classes)])
        
        plt.title(f'Vehicle Detection Analysis - {self.video_info["filename"]}', 
                 fontsize=20, fontweight='bold', pad=20)
        plt.xlabel('Vehicle Types', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Detections', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/vehicle_frequency_chart.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Diversity over time chart
        if len(self.frame_results) > 1:
            plt.figure(figsize=(15, 6))
            timestamps = [frame['timestamp'] for frame in self.frame_results]
            diversities = [frame['class_diversity'] for frame in self.frame_results]
            
            plt.plot(timestamps, diversities, marker='o', linewidth=2, markersize=4, color='#FF6B6B')
            plt.fill_between(timestamps, diversities, alpha=0.3, color='#FF6B6B')
            
            plt.title(f'Vehicle Diversity Over Time - {self.video_info["filename"]}', 
                     fontsize=16, fontweight='bold')
            plt.xlabel('Time (seconds)', fontsize=12, fontweight='bold')
            plt.ylabel('Number of Unique Vehicle Types', fontsize=12, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/diversity_timeline.png", dpi=300, bbox_inches='tight')
            plt.close()
    
    def print_comprehensive_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "="*80)
        print("COMPREHENSIVE VEHICLE DETECTION ANALYSIS REPORT 🚗")
        print("="*80)
        
        total_detections = sum(self.class_counts.values())
        total_frames = len(self.frame_results)
        
        print(f"Video Analysis Summary:")
        print(f"   • File: {self.video_info['filename']}")
        print(f"   • Duration: {self.video_info['duration']:.2f} seconds")
        print(f"   • Resolution: {self.video_info['width']}x{self.video_info['height']}")
        print(f"   • Original FPS: {self.video_info['fps']:.1f}")
        
        print(f"\nDetection Statistics:")
        print(f"   • Frames processed: {total_frames} (every 5th frame)")
        print(f"   • Total vehicle detections: {total_detections}")
        print(f"   • Average detections per frame: {total_detections/max(total_frames,1):.2f}")
        print(f"   • Unique vehicle types found: {len(self.class_counts)}")
        
        if self.class_counts:
            print(f"\nVehicle Type Distribution:")
            for vehicle_type, count in sorted(self.class_counts.items(), 
                                            key=lambda x: x[1], reverse=True):
                percentage = (count / total_detections) * 100
                bar_length = int(percentage / 2)
                bar = "█" * bar_length
                print(f"   • {vehicle_type.replace('_', ' ').title():15}: {count:3d} ({percentage:5.1f}%) {bar}")
        
        max_diversity = self.get_max_diversity_frame()
        if max_diversity:
            print(f"\n Frame with Maximum Diversity:")
            print(f"   • Frame Number: {max_diversity['frame_number']}")
            print(f"   • Timestamp: {max_diversity['timestamp']:.2f}s")
            print(f"   • Unique vehicle types: {max_diversity['class_diversity']}")
            print(f"   • Vehicle types present: {', '.join(max_diversity['unique_classes'])}")
            print(f"   • Total objects in frame: {max_diversity['total_objects']}")
        
        print("\n" + "="*80)


def main():
    print("ENHANCED MULTI-VIDEO VEHICLE DETECTION SYSTEM")
    
    detector = EnhancedMultiVideoVehicleDetectionEngine(data_folder='data')
    
    try:
        video_path = detector.select_video()
        
        if video_path is None:
            print("👋 No video selected. Exiting...")
            return
        
        print(f"\n🎬 Processing selected video: {os.path.basename(video_path)}")
        results = detector.process_video(video_path)
        
        if results is not None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_folder = f"detection_results_{video_name}"
            print(f"\nAnalysis complete! Results saved to '{output_folder}/':")
           
        
    except KeyboardInterrupt:
        print("\n\nProcess interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n Error during processing: {str(e)}")
        print("Please check your video files and try again.")


if __name__ == "__main__":
    main()  