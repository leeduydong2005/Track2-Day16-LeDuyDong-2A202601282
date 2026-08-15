# Báo cáo — Lab 16: Cloud AI Environment Setup

1. Hạ tầng AWS đã được triển khai bằng Terraform với Bastion Host và CPU Compute Node trong Private VPC; benchmark sử dụng bộ dữ liệu Credit Card Fraud gồm 284.807 bản ghi.
2. Dữ liệu được nạp trong 2,2548 giây; tập huấn luyện có 227.845 dòng và tập kiểm tra có 56.962 dòng.
3. Mô hình LightGBM được huấn luyện trong 17,3487 giây với 500 vòng lặp (best iteration = 500).
4. Mô hình đạt AUC-ROC 0,892760 và Accuracy 0,998929, cho thấy khả năng phân biệt giao dịch tốt.
5. F1-score là 0,693467; Precision 0,683168 và Recall 0,704082. Accuracy rất cao cần được diễn giải thận trọng vì dữ liệu gian lận mất cân bằng mạnh.
6. Độ trễ inference cho một dòng là 1,4682 ms; batch 1.000 dòng mất 0,025298 giây, tương ứng throughput 39.528,66 dòng/giây.
7. Kết quả cho thấy CPU node phù hợp cho benchmark LightGBM và inference batch nhẹ; file `benchmark_result.json` lưu toàn bộ số liệu chi tiết.
8. Tại thời điểm kiểm tra, AWS Billing hiển thị USD 0,00 và chưa có dữ liệu do độ trễ cập nhật (thường khoảng 24 giờ); tài khoản Free Plan sử dụng credit để bù chi phí hợp lệ.
