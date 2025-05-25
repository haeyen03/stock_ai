from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import cx_Oracle # SQLite 대신 cx_Oracle 사용
from tensorflow.keras.models import load_model # TensorFlow 2.x 이상 권장
# from sklearn.preprocessing import MinMaxScaler # 필요시 사용 (여기서는 단순 정규화 사용)
import datetime
import os

app = Flask(__name__)
print("app.py 스크립트 시작")
# -----------------------------------------------------------------------------
# 사용자 설정 영역: 자신의 Oracle DB 환경에 맞게 수정하세요.
# -----------------------------------------------------------------------------
DB_USERNAME = "system"  # <<--- 실제 사용자 이름으로 변경
DB_PASSWORD = "1234"    # <<--- 실제 비밀번호로 변경
DB_DSN = "localhost:1522/xe" # <<--- 실제 DSN으로 변경

DEFAULT_STOCK_SYMBOL = "ETH" 
SEQ_LEN = 50  # 노트북에서 사용한 sequence length

# 모델을 저장할 딕셔너리 (앱 시작 시 또는 요청 시 로드)
loaded_models = {} 
MODEL_BASE_DIR = 'models' # 노트북에서 모델을 저장한 폴더

# -----------------------------------------------------------------------------
# Oracle Instant Client 경로 설정 (필요한 경우 주석 해제 및 경로 수정)
# -----------------------------------------------------------------------------
# if os.name == 'nt': # Windows 예시
#     try:
#         # cx_Oracle.init_oracle_client(lib_dir=r"C:\oracle\instantclient_19_19") # <<--- 실제 경로로 변경
#         print("Flask 앱: Oracle Instant Client 초기화 시도 (Windows).")
#     except Exception as e:
#         print(f"Flask 앱: Oracle Instant Client 초기화 실패 (Windows): {e}")
# elif os.name == 'posix': # macOS / Linux
#     pass
# -----------------------------------------------------------------------------

def get_table_and_model_path(symbol):
    """심볼에 따른 테이블명과 모델 파일 경로를 반환합니다."""
    # 노트북에서 저장한 실제 모델 파일명과 DB 테이블명을 정확히 입력해야 합니다.
    # 모델 파일명은 노트북의 ModelCheckpoint 콜백에서 생성된 이름 중 가장 좋은 것을 사용하거나,
    # 직접 latest_..._model.h5 와 같이 변경하여 사용합니다.
    if symbol == "ETH":
        return "ETH_PRICES_DATA", os.path.join(MODEL_BASE_DIR, "latest_eth_model.h5") 
    elif symbol == "SAMSUNG_1Y":
        return "SAMSUNG_005930_KS_1Y", os.path.join(MODEL_BASE_DIR, "latest_samsung_1y_model.h5")
    elif symbol == "SAMSUNG_5Y":
        return "SAMSUNG_005930_KS_5Y", os.path.join(MODEL_BASE_DIR, "latest_samsung_5y_model.h5")
    elif symbol == "SAMSUNG_MAX":
        return "SAMSUNG_005930_KS_MAX", os.path.join(MODEL_BASE_DIR, "latest_samsung_max_model.h5")
    elif symbol == "SAMSUNG_YTD":
        return "SAMSUNG_005930_KS_YTD", os.path.join(MODEL_BASE_DIR, "latest_samsung_ytd_model.h5")
    else:
        print(f"알 수 없는 심볼: {symbol}")
        return None, None

def load_keras_model(model_path):
    """주어진 경로에서 Keras 모델을 로드합니다. 없으면 None 반환."""
    if model_path and os.path.exists(model_path):
        if model_path not in loaded_models: # 이미 로드되지 않았다면 로드
            try:
                loaded_models[model_path] = load_model(model_path)
                print(f"모델 '{model_path}' 로드 완료.")
            except Exception as e:
                print(f"모델 '{model_path}' 로드 중 오류: {e}")
                loaded_models[model_path] = None # 오류 발생 시 None으로 표시
        return loaded_models[model_path]
    else:
        print(f"경고: 모델 파일 '{model_path}'를 찾을 수 없습니다.")
        return None


def get_oracle_connection():
    """Oracle 데이터베이스 연결을 생성하고 반환합니다."""
    try:
        connection = cx_Oracle.connect(user=DB_USERNAME, password=DB_PASSWORD, dsn=DB_DSN, encoding="UTF-8")
        return connection
    except cx_Oracle.DatabaseError as e:
        error_obj, = e.args
        print(f"Oracle DB 연결 오류 (get_oracle_connection): {error_obj.code} - {error_obj.message}")
        raise
    except Exception as e:
        print(f"Oracle DB 연결 중 알 수 없는 오류 (get_oracle_connection): {e}")
        raise

def preprocess_data_for_prediction(data_df, seq_len):
    """예측을 위한 데이터를 전처리합니다 (0번째 값 기준 정규화)."""
    if data_df.empty or len(data_df) < seq_len:
        print(f"데이터 부족 (preprocess): {len(data_df)}/{seq_len}")
        return None, None

    # 노트북에서 AS로 지정한 컬럼명 사용
    if 'High' not in data_df.columns or 'Low' not in data_df.columns:
        print(f"오류: DataFrame에 'High' 또는 'Low' 컬럼이 없습니다. 현재 컬럼: {data_df.columns.tolist()}")
        return None, None
        
    high_prices = data_df['High'].values
    low_prices = data_df['Low'].values
    mid_prices = (high_prices + low_prices) / 2

    # 최근 seq_len 만큼의 데이터만 사용 (예측 입력용)
    sequence = mid_prices[-seq_len:]
    
    if len(sequence) < seq_len:
        print(f"시퀀스 데이터 부족 (preprocess): {len(sequence)}/{seq_len}")
        return None, None
        
    if sequence[0] == 0 or pd.isna(sequence[0]): # 0 또는 NaN으로 나누는 오류 방지
        print(f"경고: 정규화 기준값(sequence[0])이 0 또는 NaN입니다: {sequence[0]}")
        # 모든 값이 0이거나 NaN이면 예측 불가
        if np.all(sequence == 0) or pd.isna(sequence).all():
            return None, None
        # 첫 번째 유효한 값을 찾아 기준으로 사용하거나 다른 처리
        valid_first_val_index = np.where((sequence != 0) & (~pd.isna(sequence)))[0]
        if len(valid_first_val_index) == 0: return None, None
        first_val_for_norm = sequence[valid_first_val_index[0]]
        if first_val_for_norm == 0: return None, None # 그래도 0이면 처리 불가
    else:
        first_val_for_norm = sequence[0]

    normalized_sequence = [((float(p) / float(first_val_for_norm)) - 1) if pd.notna(p) else 0 for p in sequence] # NaN은 0으로 처리
    
    return np.array([normalized_sequence]), first_val_for_norm


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_stock_data', methods=['GET'])
def get_stock_data():
    symbol = request.args.get('symbol', DEFAULT_STOCK_SYMBOL)
    table_name, model_path_for_symbol = get_table_and_model_path(symbol)

    if not table_name:
        return jsonify({'error': '유효하지 않은 심볼입니다.'}), 400
    
    current_model = load_keras_model(model_path_for_symbol)

    conn = None
    data_df = pd.DataFrame()
    try:
        conn = get_oracle_connection()
        
        # 컬럼명 AS로 노트북과 유사하게 맞춰주기
        if "ETH" in table_name.upper():
            query = f'SELECT TRADE_DATE AS "Date", OPEN_PRICE AS "Open", HIGH_PRICE AS "High", LOW_PRICE AS "Low", CLOSE_PRICE AS "Close", TRADE_VOLUME_TEXT AS "Volume", MARKET_CAP_TEXT AS "Market_Cap" FROM {table_name} ORDER BY TRADE_DATE DESC'
        else: # 삼성전자
            query = f'SELECT TRADE_DATE AS "Date", OPEN_PRICE AS "Open", HIGH_PRICE AS "High", LOW_PRICE AS "Low", CLOSE_PRICE AS "Close", ADJ_CLOSE_PRICE AS "Adj Close", TRADE_VOLUME AS "Volume" FROM {table_name} ORDER BY TRADE_DATE DESC'

        limited_query = f"SELECT * FROM ({query}) WHERE ROWNUM <= 200" # 최근 200개
        data_df = pd.read_sql_query(limited_query, conn)
        
        if data_df.empty:
            return jsonify({'error': f'{symbol} 데이터가 DB에 없습니다.'}), 404

        data_df = data_df.iloc[::-1].reset_index(drop=True) # 시간 순 정렬 및 인덱스 리셋
        data_df['Date'] = pd.to_datetime(data_df['Date']).dt.strftime('%Y-%m-%d')
        
        # 숫자형 변환 (DB에서 이미 NUMBER형이면 대부분 float64로 오지만, 확인차원)
        numeric_cols_map = {'Open':float, 'High':float, 'Low':float, 'Close':float}
        if "ETH" in table_name.upper():
            def clean_and_convert_to_float_for_df(value):
                if pd.isna(value) or not isinstance(value, str): return float(value) if isinstance(value, (int, float)) else np.nan
                try: return float(value.replace(',', ''))
                except: return np.nan
            data_df['Volume'] = data_df['Volume'].apply(clean_and_convert_to_float_for_df)
            data_df['Market_Cap'] = data_df['Market_Cap'].apply(clean_and_convert_to_float_for_df)
        else: # 삼성
            numeric_cols_map['Adj Close'] = float
            numeric_cols_map['Volume'] = float # 삼성은 Volume이 숫자형
        
        for col, dtype in numeric_cols_map.items():
            if col in data_df.columns:
                data_df[col] = pd.to_numeric(data_df[col], errors='coerce').astype(dtype)
        
        # NaN 처리
        data_df.dropna(subset=['Date', 'High', 'Low'], inplace=True) # 예측에 필수적인 컬럼 기준

    except cx_Oracle.Error as e:
        error_obj, = e.args
        print(f"Oracle DB 오류 (get_stock_data): {error_obj.code} - {error_obj.message}")
        return jsonify({'error': f'DB 오류: {error_obj.message}'}), 500
    except Exception as e:
        print(f"초기 데이터 처리 중 오류 (get_stock_data): {e}")
        return jsonify({'error': f'처리 오류: {e}'}), 500
    finally:
        if conn:
            conn.close()
    
    if data_df.empty:
        return jsonify({'error': f'{symbol} 처리 후 데이터가 없습니다.'}), 404

    mid_prices = (data_df['High'] + data_df['Low']) / 2
    
    x_pred_normalized, first_val_for_denorm = preprocess_data_for_prediction(data_df.copy(), SEQ_LEN)
    
    prediction_result = None
    if current_model is not None and x_pred_normalized is not None and first_val_for_denorm is not None:
        try:
            predicted_normalized_price = current_model.predict(x_pred_normalized.reshape(1, SEQ_LEN, 1))[0,0]
            predicted_price = first_val_for_denorm * (1 + predicted_normalized_price)
            
            last_date_str = data_df['Date'].iloc[-1]
            last_date = datetime.datetime.strptime(last_date_str, '%Y-%m-%d')
            next_date_pred = (last_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
            
            prediction_result = {'date': next_date_pred, 'price': float(predicted_price)}
        except Exception as e:
            print(f"초기 예측 중 오류 ({symbol}): {e}")
            prediction_result = None

    return jsonify({
        'dates': data_df['Date'].tolist(),
        'prices': mid_prices.tolist(),
        'actual_highs': data_df['High'].tolist(),
        'actual_lows': data_df['Low'].tolist(),
        'initial_prediction': prediction_result,
        'symbol': symbol
    })


@app.route('/add_and_predict', methods=['POST'])
def add_and_predict():
    form_data = request.json
    symbol = form_data.get('symbol', DEFAULT_STOCK_SYMBOL)
    table_name, model_path_for_symbol = get_table_and_model_path(symbol)

    if not table_name:
        return jsonify({'error': '유효하지 않은 심볼입니다.'}), 400

    current_model = load_keras_model(model_path_for_symbol)
    if current_model is None:
         return jsonify({'error': f'{symbol} 모델을 로드할 수 없습니다.'}), 500

    conn = None
    try:
        trade_date_str = form_data.get('trade_date')
        open_price = float(form_data.get('open_price'))
        high_price = float(form_data.get('high_price'))
        low_price = float(form_data.get('low_price'))
        close_price = float(form_data.get('close_price'))
        
        trade_date = datetime.datetime.strptime(trade_date_str, '%Y-%m-%d').date()

        conn = get_oracle_connection()
        cursor = conn.cursor()

        if "ETH" in table_name.upper():
            volume_text = str(form_data.get('volume')) if form_data.get('volume') else None
            market_cap_text = str(form_data.get('market_cap')) if form_data.get('market_cap') else None
            sql_upsert = f"""
            MERGE INTO {table_name} tgt
            USING (SELECT TO_DATE(:td_str, 'YYYY-MM-DD') AS TRADE_DATE FROM dual) src
            ON (tgt.TRADE_DATE = src.TRADE_DATE)
            WHEN MATCHED THEN
                UPDATE SET OPEN_PRICE = :op, HIGH_PRICE = :hp, LOW_PRICE = :lp, CLOSE_PRICE = :cp, 
                           TRADE_VOLUME_TEXT = :vol, MARKET_CAP_TEXT = :mcap
            WHEN NOT MATCHED THEN
                INSERT (TRADE_DATE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_PRICE, 
                        TRADE_VOLUME_TEXT, MARKET_CAP_TEXT)
                VALUES (TO_DATE(:td_str, 'YYYY-MM-DD'), :op, :hp, :lp, :cp, :vol, :mcap)
            """
            cursor.execute(sql_upsert, td_str=trade_date_str, op=open_price, hp=high_price, lp=low_price, cp=close_price, 
                           vol=volume_text, mcap=market_cap_text)
        else: # 삼성전자
            adj_close_price = float(form_data.get('adj_close_price')) if form_data.get('adj_close_price') else None
            trade_volume = int(form_data.get('volume')) if form_data.get('volume') else None
            sql_upsert = f"""
            MERGE INTO {table_name} tgt
            USING (SELECT TO_DATE(:td_str, 'YYYY-MM-DD') AS TRADE_DATE FROM dual) src
            ON (tgt.TRADE_DATE = src.TRADE_DATE)
            WHEN MATCHED THEN
                UPDATE SET OPEN_PRICE = :op, HIGH_PRICE = :hp, LOW_PRICE = :lp, CLOSE_PRICE = :cp, 
                           ADJ_CLOSE_PRICE = :acp, TRADE_VOLUME = :vol
            WHEN NOT MATCHED THEN
                INSERT (TRADE_DATE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_PRICE, 
                        ADJ_CLOSE_PRICE, TRADE_VOLUME)
                VALUES (TO_DATE(:td_str, 'YYYY-MM-DD'), :op, :hp, :lp, :cp, :acp, :vol)
            """
            cursor.execute(sql_upsert, td_str=trade_date_str, op=open_price, hp=high_price, lp=low_price, cp=close_price, 
                           acp=adj_close_price, vol=trade_volume)
        
        conn.commit()
        print(f"'{trade_date_str}' 데이터 DB에 추가/업데이트 완료 ({table_name}).")

        # 예측을 위해 DB에서 최신 데이터 다시 로드 (사용자 입력 포함)
        if "ETH" in table_name.upper():
            query = f'SELECT TRADE_DATE AS "Date", OPEN_PRICE AS "Open", HIGH_PRICE AS "High", LOW_PRICE AS "Low", CLOSE_PRICE AS "Close" FROM {table_name} WHERE TRADE_DATE <= TO_DATE(:input_date, \'YYYY-MM-DD\') ORDER BY TRADE_DATE DESC'
        else: # 삼성
            query = f'SELECT TRADE_DATE AS "Date", OPEN_PRICE AS "Open", HIGH_PRICE AS "High", LOW_PRICE AS "Low", CLOSE_PRICE AS "Close", ADJ_CLOSE_PRICE AS "Adj_Close" FROM {table_name} WHERE TRADE_DATE <= TO_DATE(:input_date, \'YYYY-MM-DD\') ORDER BY TRADE_DATE DESC'
        
        limited_query = f"SELECT * FROM ({query}) WHERE ROWNUM <= {SEQ_LEN + 5}"
        data_df_updated = pd.read_sql_query(limited_query, conn, params={'input_date': trade_date_str})

        if data_df_updated.empty or len(data_df_updated) < SEQ_LEN:
            return jsonify({'error': 'DB 업데이트 후 예측 데이터 로드 실패 또는 부족.'}), 400
            
        data_df_updated = data_df_updated.iloc[::-1].reset_index(drop=True)
        data_df_updated['Date'] = pd.to_datetime(data_df_updated['Date']).dt.strftime('%Y-%m-%d')
        
        # 숫자형 변환
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in data_df_updated.columns:
                 data_df_updated[col] = pd.to_numeric(data_df_updated[col], errors='coerce').astype(float)
        if 'Adj Close' in data_df_updated.columns: # 삼성전자용
            data_df_updated['Adj Close'] = pd.to_numeric(data_df_updated['Adj Close'], errors='coerce').astype(float)
        
        data_df_updated.dropna(subset=['Date', 'High', 'Low'], inplace=True)
        if data_df_updated.empty or len(data_df_updated) < SEQ_LEN:
             return jsonify({'error': 'NaN 제거 후 데이터 부족.'}), 400


        x_pred_normalized, first_val_for_denorm = preprocess_data_for_prediction(data_df_updated.copy(), SEQ_LEN)

        if x_pred_normalized is None or first_val_for_denorm is None:
            return jsonify({'error': '업데이트된 데이터로 전처리 실패.'}), 400

        predicted_normalized_price = current_model.predict(x_pred_normalized.reshape(1, SEQ_LEN, 1))[0,0]
        predicted_price_val = first_val_for_denorm * (1 + predicted_normalized_price)
        
        predicted_for_date_obj = datetime.datetime.strptime(trade_date_str, '%Y-%m-%d')
        final_prediction_date_str = (predicted_for_date_obj + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        
        updated_mid_prices = (data_df_updated['High'] + data_df_updated['Low']) / 2

        return jsonify({
            'message': '데이터 추가 및 예측 성공',
            'predicted_date': final_prediction_date_str,
            'predicted_price': float(predicted_price_val),
            'updated_dates': data_df_updated['Date'].tolist(),
            'updated_prices': updated_mid_prices.tolist(),
            'user_input_date': trade_date_str, # 사용자가 입력한 날짜
            'user_input_mid_price': (high_price + low_price) / 2 # 사용자가 입력한 날짜의 중간 가격
        })

    except cx_Oracle.Error as e:
        error_obj, = e.args
        print(f"Oracle DB 작업 중 오류 (add_and_predict): {error_obj.code} - {error_obj.message}")
        if conn: conn.rollback()
        return jsonify({'error': f'DB 오류: {error_obj.message}'}), 500
    except ValueError as e:
        print(f"데이터 변환 오류 (add_and_predict): {e}")
        return jsonify({'error': f'입력 데이터 형식 오류: {e}'}), 400
    except Exception as e:
        print(f"새 데이터 추가 및 예측 중 오류 (add_and_predict): {e}")
        if conn: conn.rollback()
        return jsonify({'error': str(e)}), 500
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    if not os.path.exists(MODEL_BASE_DIR):
        os.makedirs(MODEL_BASE_DIR)
        print(f"알림: '{MODEL_BASE_DIR}' 폴더가 없어 새로 생성했습니다. 학습된 모델 파일을 여기에 넣어주세요.")
    
    # (Instant Client 초기화 코드 필요시 여기에 추가 - populate_oracle_db.py와 동일)
    app.run(debug=True)