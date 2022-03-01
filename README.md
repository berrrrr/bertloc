# BertLoc: Duplicate Location Record Detection in a Large-Scale Location Dataset
Sujin Park, Sangwon Lee, and Simon S. Woo
SAC: The 36th ACM/SIGAPP Symposium On Applied Computing, Gwangju, Korea, 2021.

## abstract
In this work, we propose BertLoc, a novel deep learning-based architecture to detect the duplicate location represented in different ways (e.g., Cafe vs. Coffee House) and effectively merge them into a single and consistent location record. BertLoc is based on Multilingual Bert Model followed by BiLSTM and CNN to effectively compare and determine whether given location strings are the same location or not. We evaluate BertLoc trained with more than half a million location data used in real service in South Korea and compare the results with other popular baseline methods. Our experimental results show that BertLoc outperforms other popular baseline methods with 0.952 F1-score, and shows great promise in detecting duplicate records in a large-scale location dataset
