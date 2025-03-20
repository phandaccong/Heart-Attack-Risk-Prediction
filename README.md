# Phân tích dữ liệu về nguy cơ mắc bênh nhồi máu cơ tim theo nhóm tuổi ( 18 - 90 ) 
## Mô tả: 
  Theo [Link](https://tamanhhospital.vn/nhoi-mau-co-tim/#:~:text=T%E1%BA%A1i%20Vi%E1%BB%87t%20Nam%2C%20theo%20th%E1%BB%91ng,do%20nh%E1%BB%93i%20m%C3%A1u%20c%C6%A1%20tim.)
  
  _Tại Việt Nam, theo thống kê của Bộ Y tế, có hơn ***200.000*** người tử vong do bệnh tim mạch trong năm ***2023***, chiếm ***33%*** tổng số ca tử vong. Trong đó, có tới ***85%*** là do nhồi máu cơ tim._

Dự án phân tích dữ liệu y tế để nhận diện các yếu tố ảnh hưởng đến nguy cơ mắc bệnh nhồi máu cơ tim. Từ đó xây dựng mô hình dự đoán giúp hỗ trợ cảnh báo sớm và đề xuất các giải pháp phòng tránh cho từng nhóm đối tượng theo độ tuổi và chỉ số sức khỏe.

## Dữ liệu 
- Nguồn: Kaggle / Tự tổng hợp  
- Số lượng: Khoảng ~2.000 
- Các trường dữ liệu chính:
  - Tuổi, Giới tính  
  - Huyết áp tâm trương/ tâm thu
  - Mức cholesterol  
  - Nhịp tim tối đa
  - ....
  - Nguy cơ nhồi máu cơ tim (0/1)
##  Công cụ & Công nghệ
- Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)  
- Power BI để trực quan hóa dữ liệu  
- Thuật toán: LightGBM
## Quy trình xử lý và phân tích
- Làm sạch dữ liệu: Xử lý giá trị thiếu, loại bỏ dữ liệu trùng lặp  
- Phân tích dữ liệu khám phá (EDA):
  - Vẽ biểu đồ 
  - Tương quan giữa các biến    
## Xây dựng mô hình
- Thử nghiệm nhiều mô hình (Logistic Regression, Random Forest,Dêcnsion Tree, LightGBM)  
- Lựa chọn LightGBM + GridSearch để tìm các siêu tham số cùng hiệu suất và độ chính xác cao  
- Đánh giá mô hình: F1-score đạt **71%**
## Dashboard trực quan
[Reports](images/DAT111_DP03_FINAL.pdf)

<a href="images/DAT111_DP03_FINAL.pdf">
  <img src="images/pdf-preview.png" width="200" />
</a>

## Khuyến nghị
- Thực hiện kiểm tra sức khỏe định kỳ cho nhóm tuổi nguy cơ cao  
- Theo dõi kỹ chỉ số cholesterol và thalassemia  
- Mở rộng thu thập dữ liệu từ nhiều bệnh viện để tăng độ chính xác và phổ quát của mô hình  



  
