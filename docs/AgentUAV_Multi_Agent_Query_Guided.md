# AgentUAV: Multi-Agent Query-Guided UAV Target Detection with Super-Resolution and SAHI

**Authors:**
- N. Vigneshwaran (Assistant Professor)
- O. Vikram Rajpurohit
- M. Tharanidharan
- S.K. Srinidhi Ujaini
(Department of Artificial Intelligence and Data Science, Kongu Engineering College, Erode, Tamil Nadu, India)

## Abstract
The applications of UAV-based surveillance, search, and rescue, and aerial monitoring all demand the robust detection of targets that may be difficult to recognize due to their small size long distance partial visibility, or low image quality, among others. While conventional object detection models are mostly dependent on the predefined object categories, they may run into challenges when the target or scene category varies. Open-vocabulary detection can recognize the category of targets using the query text, but applying the same processing approach to all UAV images can mean unnecessary cost of computation or underprocessing of complex targets. To overcome these deficiencies, we propose AgentUAV-a multi-agent query-guided UAV target detection system, which combines open-vocabulary detection with Slicing Aided Hyper Inference (SAHI), Super-Resolution, and adaptive processing. The system takes as input a UAV image or video frame and a natural-language target query and endeavors to first identify the requested target with YOLO-World. The detected results are then analyzed based on target types and visual contexts to determine whether further processing is needed. Use of the SAHI based sliced inference is extended for small far away targets, whereas Super-Resolution enhancement is used to address low resolution or degraded parts of the image. Multiple adaptive specialized agents work together for query understanding, target detection, analysis, and verification, and adaptive decision-making to produce the final result of target detection. The comprehensive system is evaluated in multiple detection set-ups to investigate the advantage of sliced inference, image enhancement and adaptive multi-agent processing. The results show that combining these approaches can positively benefit the target detection coverage in challenging UAV scenarios in particular for small and hard to detect targets. The proposed AgentUAV setup brings a highly flexible approach for query-based UAV target detection, by selecting different approaches per the input image and detected target types.

**Keywords** — UAV Target Detection, Multi-Agent Framework, Query-Guided Detection, Open-Vocabulary Object Detection, SAHI-Based Sliced Inference, Super-Resolution, Adaptive Target Detection.

*(Note: This is the extracted text from the uploaded PDF document)*
