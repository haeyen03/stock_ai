import pandas as pd
import cx_Oracle
import os
from datetime import datetime
import glob # 파일 목록을 가져오기 위해 추가
import numpy as np 

# --- Oracle DB 접속 정보 ---
DB_USERNAME = "system"
DB_PASSWORD = "1234"
DB_DSN = "localhost:1522/xe"

BASE_PROJECT_DIR_NAME = "stock_crypto_price_prediction-master" 
DATASET_DIR = os.path.join(BASE_PROJECT_DIR_NAME, 'dataset')
# 만약 이 스크립트가 stock_crypto_price_prediction-master 폴더 안에 있고, 
# dataset 폴더도 그 안에 있다면 DATASET_DIR = 'dataset' 으로 변경하세요.

# -----------------------------------------------------------------------------
# Oracle Instant Client 경로 설정 (필요한 경우 주석 해제 및 경로 수정)
# -----------------------------------------------------------------------------
# if os.name == 'nt': # Windows 예시
#     try:
#         # 예: cx_Oracle.init_oracle_client(lib_dir=r"C:\oracle\instantclient_19_19")
#         # cx_Oracle.init_oracle_client(lib_dir=r"YOUR_WINDOWS_INSTANT_CLIENT_PATH_HERE")
#         print("Oracle Instant Client 초기화 시도 (Windows). 실제 경로는 코드 내에서 수정 필요.")
#     except Exception as e:
#         print(f"Oracle Instant Client 초기화 실패 (Windows): {e}")
#         print("Oracle Instant Client가 설치되어 있고 PATH에 등록되어 있는지, 또는 위 경로가 올바른지 확인하세요.")
# elif os.name == 'posix': # macOS / Linux 예시
#     # macOS: 터미널에서 'export DYLD_LIBRARY_PATH=/path/to/instantclient:$DYLD_LIBRARY_PATH' 실행
#     # Linux: 터미널에서 'export LD_LIBRARY_PATH=/path/to/instantclient:$LD_LIBRARY_PATH' 실행
#     # 또는 아래 init_oracle_client 사용 (덜 일반적)
#     # try:
#     #     # 예: cx_Oracle.init_oracle_client(lib_dir="/opt/oracle/instantclient_19_8")
#     #     # cx_Oracle.init_oracle_client(lib_dir="YOUR_MACOS_LINUX_INSTANT_CLIENT_PATH_HERE")
#     #     print("Oracle Instant Client 초기화 시도 (macOS/Linux). 실제 경로는 코드 내에서 수정 필요.")
#     # except Exception as e:
#     #     print(f"Oracle Instant Client 초기화 실패 (macOS/Linux): {e}")
#     #     print("Oracle Instant Client가 설치되어 있고 (DY)LD_LIBRARY_PATH에 등록되어 있는지, 또는 위 경로가 올바른지 확인하세요.")
#     pass
# -----------------------------------------------------------------------------

def get_oracle_connection():
    """Oracle 데이터베이스 연결을 생성하고 반환합니다."""
    try:
        connection = cx_Oracle.connect(user=DB_USERNAME, password=DB_PASSWORD, dsn=DB_DSN, encoding="UTF-8")
        print(f"Oracle DB ({DB_DSN})에 성공적으로 연결되었습니다.")
        return connection
    except cx_Oracle.DatabaseError as e:
        error_obj, = e.args
        print(f"Oracle DB 연결 오류: {error_obj.code} - {error_obj.message}")
        print(f"DSN: {DB_DSN}, User: {DB_USERNAME}")
        print("DB 접속 정보, 리스너 상태, Instant Client 설정을 확인하세요.")
        raise
    except Exception as e:
        print(f"Oracle DB 연결 중 알 수 없는 오류: {e}")
        raise

def parse_samsung_date(date_str):
    """삼성전자 CSV의 날짜 문자열(예: '2018-01-02')을 Python date 객체로 변환합니다."""
    if pd.isna(date_str): return None
    try:
        return pd.to_datetime(date_str).to_pydatetime().date()
    except Exception as e:
        print(f"경고: 삼성전자 날짜 파싱 실패 '{date_str}': {e}. 해당 행을 건너뜁니다.")
        return None

def parse_eth_date(date_str):
    """ETH CSV의 날짜 문자열(예: '31.Oct.17')을 Python date 객체로 변환합니다."""
    if pd.isna(date_str): return None
    try:
        return datetime.strptime(str(date_str), '%d.%b.%y').date()
    except ValueError:
        try:
            dt_obj_pandas = pd.to_datetime(date_str)
            if pd.isna(dt_obj_pandas):
                print(f"경고: ETH 날짜 파싱 실패 (NaT 반환) '{date_str}'. 해당 행을 건너뜁니다.")
                return None
            return dt_obj_pandas.to_pydatetime().date()
        except Exception as e:
            print(f"경고: ETH 날짜 파싱 실패 (두 번째 시도) '{date_str}': {e}. 해당 행을 건너뜁니다.")
            return None

def populate_table_from_csv(connection, csv_file_path, table_name):
    """주어진 CSV 파일의 데이터를 지정된 Oracle 테이블에 삽입합니다."""
    cursor = None
    print(f"\n--- 테이블 '{table_name}' 처리 시작: {os.path.basename(csv_file_path)} ---")

    try:
        cursor = connection.cursor()
        data_df = pd.read_csv(csv_file_path)

        # 공통적으로 비어있는 문자열이나 'null' 문자열을 NaN으로 처리
        data_df.replace(['', 'null', 'NaN', 'NAN', 'nan'], np.nan, inplace=True)

        records_to_insert = []
        sql = ""

        if "005930.KS" in csv_file_path.upper(): # 삼성전자 CSV 파일들
            # NaN 값을 Python None으로 먼저 변환
            data_df = data_df.where(pd.notnull(data_df), None)

            for index, row in data_df.iterrows():
                trade_date = parse_samsung_date(row.get('Date'))
                if trade_date is None:
                    continue # 날짜 파싱 실패 시 해당 행 건너뛰기

                # 모든 가격 및 거래량 컬럼을 숫자로 변환 (실패 시 None)
                open_price = pd.to_numeric(row.get('Open'), errors='coerce')
                high_price = pd.to_numeric(row.get('High'), errors='coerce')
                low_price = pd.to_numeric(row.get('Low'), errors='coerce')
                close_price = pd.to_numeric(row.get('Close'), errors='coerce')
                adj_close_price = pd.to_numeric(row.get('Adj Close'), errors='coerce')
                trade_volume = pd.to_numeric(row.get('Volume'), errors='coerce')

                record = (
                    trade_date,
                    None if pd.isna(open_price) else float(open_price),
                    None if pd.isna(high_price) else float(high_price),
                    None if pd.isna(low_price) else float(low_price),
                    None if pd.isna(close_price) else float(close_price),
                    None if pd.isna(adj_close_price) else float(adj_close_price),
                    None if pd.isna(trade_volume) else int(trade_volume) if pd.notna(trade_volume) else None
                )
                records_to_insert.append(record)
            
            sql = f"""INSERT INTO {table_name} (TRADE_DATE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_PRICE, ADJ_CLOSE_PRICE, TRADE_VOLUME) 
                       VALUES (:1, :2, :3, :4, :5, :6, :7)"""

        elif "ETH.CSV" in csv_file_path.upper(): # ETH CSV 파일
            data_df = data_df.where(pd.notnull(data_df), None)

            for index, row in data_df.iterrows():
                trade_date = parse_eth_date(row.get('Date'))
                if trade_date is None:
                    continue
                
                open_price = pd.to_numeric(str(row.get('Open')).replace(',', ''), errors='coerce')
                high_price = pd.to_numeric(str(row.get('High')).replace(',', ''), errors='coerce')
                low_price = pd.to_numeric(str(row.get('Low')).replace(',', ''), errors='coerce')
                close_price = pd.to_numeric(str(row.get('Close')).replace(',', ''), errors='coerce')

                volume_text = str(row.get('Volume')) if pd.notnull(row.get('Volume')) else None
                market_cap_text = str(row.get('Market Cap')) if pd.notnull(row.get('Market Cap')) else None
                
                record = (
                    trade_date,
                    None if pd.isna(open_price) else float(open_price),
                    None if pd.isna(high_price) else float(high_price),
                    None if pd.isna(low_price) else float(low_price),
                    None if pd.isna(close_price) else float(close_price),
                    volume_text,
                    market_cap_text
                )
                records_to_insert.append(record)

            sql = f"""INSERT INTO {table_name} (TRADE_DATE, OPEN_PRICE, HIGH_PRICE, LOW_PRICE, CLOSE_PRICE, TRADE_VOLUME_TEXT, MARKET_CAP_TEXT) 
                       VALUES (:1, :2, :3, :4, :5, :6, :7)"""
        else:
            print(f"알 수 없는 CSV 파일 형식 또는 처리 규칙 없음: {os.path.basename(csv_file_path)}")
            return

        # 기존 데이터 삭제
        try:
            cursor.execute(f"DELETE FROM {table_name}")
            print(f"테이블 '{table_name}'의 기존 데이터 삭제 완료.")
        except cx_Oracle.Error as e:
            error_obj, = e.args
            if error_obj.code == 942: # ORA-00942: table or view does not exist
                 print(f"경고: 테이블 '{table_name}'이(가) DB에 존재하지 않아 삭제를 건너뜁니다. SQL Developer에서 테이블을 먼저 생성해주세요.")
                 return # 테이블이 없으면 삽입도 불가능하므로 함수 종료
            else:
                print(f"기존 데이터 삭제 중 오류 ({table_name}): {error_obj.code} - {error_obj.message}")
                raise # 다른 DB 오류는 다시 발생시켜 처리 중단

        if not records_to_insert:
            print(f"'{os.path.basename(csv_file_path)}' 파일에서 삽입할 유효한 데이터가 없습니다.")
            return
            
        cursor.executemany(sql, records_to_insert, batcherrors=True)
        
        batch_errors_occurred = False
        for error_obj_detail in cursor.getbatcherrors():
            batch_errors_occurred = True
            problem_row_index = error_obj_detail.offset
            # records_to_insert가 비어있지 않음을 위에서 확인했으므로 안전하게 접근
            problem_data = records_to_insert[problem_row_index] if problem_row_index < len(records_to_insert) else "N/A"
            print(f"DB 삽입 오류 (파일: {os.path.basename(csv_file_path)}, 테이블: {table_name}, CSV 행 약 {problem_row_index + 2}): {error_obj_detail.message} - 데이터: {problem_data}")


        if not batch_errors_occurred:
            connection.commit()
            print(f"'{os.path.basename(csv_file_path)}'의 데이터가 '{table_name}' 테이블에 성공적으로 삽입되었습니다.")
        else:
            print(f"'{os.path.basename(csv_file_path)}' 데이터 삽입 중 일부 오류 발생. 롤백합니다.")
            connection.rollback()

    except pd.errors.EmptyDataError:
        print(f"오류: CSV 파일 '{os.path.basename(csv_file_path)}'가 비어있습니다.")
    except FileNotFoundError:
        print(f"오류: CSV 파일 '{csv_file_path}'를 찾을 수 없습니다.")
    except cx_Oracle.Error as error:
        error_obj, = error.args
        print(f"Oracle DB 작업 중 오류 ({table_name}): {error_obj.code} - {error_obj.message}")
        if connection:
            try:
                connection.rollback()
            except cx_Oracle.Error as rb_error:
                print(f"롤백 중 추가 오류 발생: {rb_error}")
    except Exception as e:
        print(f"데이터 처리 중 일반 오류 발생 ({os.path.basename(csv_file_path)}): {e}")
        if connection:
            try:
                connection.rollback()
            except cx_Oracle.Error as rb_error:
                print(f"롤백 중 추가 오류 발생: {rb_error}")
    finally:
        if cursor:
            cursor.close()

def main_populate():
    """데이터셋 폴더의 모든 CSV를 Oracle DB의 해당 테이블로 이전합니다."""
    if not os.path.exists(DATASET_DIR):
        print(f"오류: '{DATASET_DIR}' 폴더를 찾을 수 없습니다. 스크립트 실행 위치 또는 DATASET_DIR 변수를 확인하세요.")
        return

    conn = None
    try:
        conn = get_oracle_connection()
        
        csv_files = glob.glob(os.path.join(DATASET_DIR, "*.csv"))
        if not csv_files:
            print(f"'{DATASET_DIR}' 폴더에 CSV 파일이 없습니다.")
            return

        for csv_file in csv_files:
            file_name_only = os.path.basename(csv_file).upper() # 비교를 위해 대문자로
            
            # 테이블 이름을 SQL Developer에서 생성한 이름과 정확히 일치시킵니다 (Oracle은 기본적으로 대문자)
            if "005930.KS_1Y.CSV" == file_name_only:
                populate_table_from_csv(conn, csv_file, "SAMSUNG_005930_KS_1Y")
            elif "005930.KS_5Y.CSV" == file_name_only:
                populate_table_from_csv(conn, csv_file, "SAMSUNG_005930_KS_5Y")
            elif "005930.KS_MAX.CSV" == file_name_only:
                populate_table_from_csv(conn, csv_file, "SAMSUNG_005930_KS_MAX")
            elif "005930.KS_YTD.CSV" == file_name_only:
                populate_table_from_csv(conn, csv_file, "SAMSUNG_005930_KS_YTD")
            elif "ETH.CSV" == file_name_only:
                populate_table_from_csv(conn, csv_file, "ETH_PRICES_DATA")
            else:
                print(f"처리 규칙이 없는 CSV 파일 (건너뜁니다): {os.path.basename(csv_file)}")
                
    except cx_Oracle.Error:
        # get_oracle_connection에서 이미 오류 메시지 출력되었을 것임
        print("DB 연결 실패로 작업을 중단합니다.")
    except Exception as e:
        print(f"전체 DB populate 작업 중 예상치 못한 오류 발생: {e}")
    finally:
        if conn:
            conn.close()
            print("\nOracle DB 연결이 최종적으로 닫혔습니다.")

if __name__ == '__main__':
    print("데이터베이스 채우기 스크립트 시작...")
    main_populate()
    print("\n데이터베이스 채우기 스크립트 종료.")