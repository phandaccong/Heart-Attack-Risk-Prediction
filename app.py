import streamlit as st
import pandas as pd
import joblib as jb
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.utils import check_X_y
from pymongo.mongo_client import MongoClient

# model 
column = ['Age',
 'Cholesterol',
 'Heart Rate',
 'Diabetes',
 'Family History',
 'Smoking',
 'Obesity',
 'Alcohol Consumption',
 'Exercise Hours Per Week',
 'Diet',
 'Previous Heart Problems',
 'Medication Use',
 'Stress Level',
 'Income',
 'BMI',
 'Triglycerides',
 'Physical Activity Days Per Week',
 'Sleep Hours Per Day',
 'Systolic_BP',
 'Diastolic_BP',
 'Active Hours Day',
 'Sex_Female',
 'Sex_Male',
 'Country_Argentina',
 'Country_Australia',
 'Country_Brazil',
 'Country_Canada',
 'Country_China',
 'Country_Colombia',
 'Country_France',
 'Country_Germany',
 'Country_India',
 'Country_Italy',
 'Country_Japan',
 'Country_New Zealand',
 'Country_Nigeria',
 'Country_South Africa',
 'Country_South Korea',
 'Country_Spain',
 'Country_Thailand',
 'Country_United Kingdom',
 'Country_United States',
 'Country_Vietnam']


def save_data_mongodb(df):
    uri = "mongodb+srv://phandaccong:3103@tiktokcommnet.6ep1ywo.mongodb.net/?retryWrites=true&w=majority&tlsAllowInvalidCertificates=true&appName=tiktokcomment"
    client = MongoClient(uri)
    db = client['DataHeart_DB']
    data = db['DataHeart_new']
    data.insert_one(df)
    print('Đã lưu dữ liệu ')


# Tiêu đề ứng dụng
st.title('Heart Attack Risk Predict')
st.write('Chào bạn đến với nhóm DP-03!')

# Nhập tên
name = st.text_input("Nhập vào tên của bạn:")
countries = ['France', 'Argentina', 'Japan', 'Australia', 'New Zealand', 
             'Brazil', 'United Kingdom', 'India', 'Colombia', 'Canada',
             'South Africa', 'Spain', 'Thailand', 'Vietnam', 'China',
             'South Korea', 'United States', 'Nigeria', 'Italy', 'Germany']
if name:
    st.title(f'Xin chào {name}, mời bạn nhập vào các trường sau:')
    # Các trường nhập dữ liệu
    age = st.number_input("Nhập vào tuổi của bạn (age) :", min_value=18, max_value=90)
    sex = st.selectbox("Giới tính của bạn (sex) :", ["Male", "Female"])
    chol = st.number_input("Nhập vào mức cholesterol của bạn (mg/dL):", min_value=0, max_value = 900 , value = 300)
    heart = st.number_input("Nhập vào nhịp tim (bpm):", min_value=10, max_value=500 , value = 80)
    diabetes = st.selectbox("Bạn có bị tiểu đường không(0 : không , 1: có) ? ", ["0", "1"])
    family = st.selectbox("Người nhà bạn có mắc bệnh tim không (0 : không , 1: có) ?", ["0", "1"])
    smoking = st.selectbox("Bạn có hút thuốc không (0 : không , 1: có) ?", ["0", "1"])
    obesity = st.selectbox("Nhập vào obesity: ", ['0' , '1'])
    alcohol = st.selectbox("Co uong ruou bia khong: (0 : không , 1: có)", ['0' , '1'])
    exercise = st.number_input("Số giờ tập thể dục mỗi tuần:", min_value=0.0 , value = 0.0)
    diet = st.selectbox("Chế độ ăn uống của bạn:", ["Healthy","Average","Unhealthy"])
    previous_problems = st.selectbox("Bạn từng có vấn đề tim mạch trước đây không (0 : không , 1: có) ?", ["0", "1"])
    medication = st.selectbox("Bạn có sử dụng thuốc điều trị không (0 : không , 1: có) ?", ["0", "1"])
    stress = st.number_input("Mức độ căng thẳng (1-10):", min_value=1, max_value=10 , value = 1)
    income = st.number_input("Thu nhập trung bình hàng tháng (USD):", min_value=1000.0 , value = 1000.0)
    bmi = st.number_input("Nhap vao chỉ số BMI" , min_value= 10.0 , max_value=999.0 , value = 25.0)
    triglycerides = st.number_input("Triglycerides (mg/dL):", min_value= 0.0 , max_value = 1000.0 , value = 500.0)
    physical_activity = st.number_input("Số ngày hoạt động thể chất mỗi tuần:", min_value=0, max_value=7 , value = 0)
    sleep_hours = st.number_input("Số giờ ngủ mỗi ngày:", min_value=0, max_value=24 , value = 10)
    country = st.selectbox("Chọn quốc gia của bạn", countries)
    systolic_bp = st.number_input("Huyết áp tâm thu (mmHg):", min_value=100, max_value = 900 , value = 200)
    diastolic_bp = st.number_input("Huyết áp tâm trương (mmHg):", min_value=0, max_value=900 , value = 100 )
    active_hours = st.number_input("Số giờ hoạt động mỗi ngày:", min_value=0, max_value=24 , value = 20)

    # Kiểm tra dữ liệu đã nhập
    if st.button("Dự đoán nguy cơ"):
        missing_data = []
        # Kiểm tra từng trường
        if not age:
            missing_data.append("Tuổi")
        if not chol:
            missing_data.append("Cholesterol")
        if not heart:
            missing_data.append("Nhịp tim")
        if not obesity:
            missing_data.append("Béo phì (BMI)")
        if not alcohol:
            missing_data.append("Tiêu thụ rượu bia")
        if not exercise:
            missing_data.append("Tập thể dục")
        if not stress:
            missing_data.append("Căng thẳng")
        if not income:
            missing_data.append("Thu nhập")
        if not triglycerides:
            missing_data.append("Triglycerides")
        if not physical_activity:
            missing_data.append("Hoạt động thể chất")
        if not sleep_hours:
            missing_data.append("Giờ ngủ")
        if not systolic_bp:
            missing_data.append("Huyết áp tâm thu")
        if not diastolic_bp:
            missing_data.append("Huyết áp tâm trương")
        if not active_hours:
            missing_data.append("Giờ hoạt động mỗi ngày")
        if not country:
            missing_data.append("Quốc gia")

        # Kiểm tra trường chọn
        if sex == "":
            missing_data.append("Giới tính")
        if diabetes == "":
            missing_data.append("Tiểu đường")
        if family == "":
            missing_data.append("Tiền sử gia đình")
        if smoking == "":
            missing_data.append("Hút thuốc")
        if diet == "":
            missing_data.append("Chế độ ăn uống")
        if previous_problems == "":
            missing_data.append("Vấn đề tim trước đây")
        if medication == "":
            missing_data.append("Sử dụng thuốc")
        if float(systolic_bp) < float(diastolic_bp):
            st.missing_data.append("loi systolic < diastolic")
        # Xử lý kết quả
        if missing_data:
            st.error(f"Bạn cần nhập đầy đủ thông tin cho các trường sau: {', '.join(missing_data)}")
        
        else:
            st.success("Thông tin của bạn đã được ghi nhận! Sẵn sàng dự đoán.")
            st.write(f"Thông tin của {name} đã được ghi nhận:")
                        
            keywords = {'Average':0, 'Unhealthy':1, 'Healthy':2}
            diet_again = keywords[diet]
            a = {
                'name' : str(name),
                'Age': int(age),
                'Cholesterol' : float(chol),
                'Heart Rate' : int(heart),
                'Diabetes' : int(diabetes),
                'Family History' : int(family),
                'Smoking': int(smoking),
                'Obesity': float(obesity),
                'Alcohol Consumption': float(alcohol),
                'Exercise Hours Per Week' : float(exercise),
                'Diet' : diet_again,
                'Previous Heart Problems': int(previous_problems),
                'Medication Use' : int(medication),
                'Stress Level': int(stress),
                'Income' : float(income),
                'BMI' : float(bmi),
                'Triglycerides' : float(triglycerides),
                'Physical Activity Days Per Week' : float(physical_activity),
                'Sleep Hours Per Day' : int(sleep_hours),
                'Systolic_BP' : int(systolic_bp),
                'Diastolic_BP' : int(diastolic_bp),
                'Active Hours Day': float(active_hours),
                'Sex_Female' : 1 if sex == "Female" else 0,
                'Sex_Male' : 1 if sex == "Male" else 0,
                'Country_Argentina' : 1 if country == "Argentina" else 0,
                'Country_Australia' : 1 if country == "Australia" else 0,
                'Country_Brazil' : 1 if country == "Brazil" else 0,
                'Country_Canada' : 1 if country == "Canada" else 0,
                'Country_China' : 1 if country == "China" else 0,
                'Country_Colombia' : 1 if country == "Colombia" else 0,
                'Country_France' : 1 if country == "France" else 0,
                'Country_Germany' : 1 if country == "Germany" else 0,
                'Country_India' : 1 if country == "India" else 0,
                'Country_Italy' : 1 if country == "Italy" else 0,
                'Country_Japan' : 1 if country == "Japan" else 0,
                'Country_New Zealand' : 1 if country == "New Zealand" else 0,
                'Country_Nigeria' : 1 if country == "Nigeria" else 0,
                'Country_South Africa' : 1 if country == "South Africa" else 0,
                'Country_South Korea' : 1 if country == "South Korea" else 0,
                'Country_Spain' : 1 if country == "Spain" else 0,
                'Country_Thailand' : 1 if country == "Thailand" else 0,
                'Country_United Kingdom' : 1 if country == "United Kingdom" else 0,
                'Country_United States' : 1 if country == "United States" else 0,
                'Country_Vietnam' : 1 if country == "Vietnam" else 0
}           
            save_data_mongodb(a)
            df = pd.DataFrame(a , index = [0])
            col_nunique = ['Age', 'Cholesterol', 'Heart Rate', 'Exercise Hours Per Week',
       'Stress Level', 'Income', 'BMI', 'Triglycerides',
       'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Systolic_BP',
       'Diastolic_BP', 'Active Hours Day']

            
            df = df.drop('name' , axis =1 )
            col_scaler = df.loc[: , col_nunique ].columns
            
            scaler = jb.load("scaler_lgb.pkl")
            df[col_scaler] = scaler.transform(df[col_scaler])
            df = np.round(df , 5)
            
            df_input = df.reindex(columns=column, fill_value=0)
            
            # lựa chọn model 
            model = jb.load("model_lgb.pkl")
            
            y_predict = int(model.predict(df_input))
            
            if y_predict == 0:
                st.title("chúc mừng Bạn không bị mắc bệnh tim 😁😁🌲")
            else :
                st.title("Bạn nên cẩn thận hơn vì bạn có nguy cơ mắc bệnh tim 🤷‍♀️🤷‍♀️🤦‍♂️🌲")
            st.write("⚠️⚠️Cảnh báo model chỉ dự đoán được chính xác 72% thôi chúng ta nên phòng tránh và chú ý tới sức khỏe hơn 💕💕💕❤️")
            st.write(a)
