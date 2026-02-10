import streamlit as st
import pandas as pd
from docx import Document
import io
import os
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

# ==========================================
# 1. 헬퍼 함수: 파일 처리 및 스키마 변환 (수정됨)
# ==========================================

def read_docx(file):
    """docx 파일을 읽어 텍스트로 반환"""
    try:
        document = Document(io.BytesIO(file.read()))
        return "\n".join([p.text for p in document.paragraphs if p.text.strip()])
    except Exception as e:
        st.error(f"파일을 읽는 중 오류가 발생했습니다: {e}")
        return ""

def parse_format_to_schema(format_text):
    """
    [수정됨] LangChain 호환 JSON Schema 생성
    OpenAI API Wrapper를 제거하고 순수 Schema만 반환합니다.
    """
    properties = {}
    required_fields = []
    
    lines = format_text.split('\n')
    for line in lines:
        if ":" in line:
            key, value_desc = line.split(":", 1)
            key = key.strip()
            value_desc = value_desc.strip().lower()
            
            # 타입 매핑
            if "int" in value_desc:
                dtype = "integer"
            elif "float" in value_desc:
                dtype = "number"
            elif "bool" in value_desc:
                dtype = "boolean"
            else:
                dtype = "string"
            
            # description에 컨텍스트(설명)를 포함시켜 GPT가 더 잘 이해하도록 함
            properties[key] = {
                "type": dtype, 
                "description": f"Extract {key}. Context hint: {value_desc}"
            }
            required_fields.append(key)
    
    # LangChain이 좋아하는 'title'과 'description'이 포함된 순수 JSON Schema 구조
    schema = {
        "title": "MedicalDataExtraction",
        "description": "Extracts structured clinical data from the report based on the provided schema.",
        "type": "object",
        "properties": properties,
        "required": required_fields,
    }
    return schema

# ==========================================
# 2. 메인 UI (Streamlit) - 디자인 유지
# ==========================================

def main():
    st.title("GPT Information Extractor v2.1")
    st.markdown("<small>Fixed Schema Compatibility</small>", unsafe_allow_html=True)
    
    # 1. API Key 입력
    api_key = st.text_input("Enter GPT API Key", type="password", help="OpenAI API Key를 입력하세요.")
    
    # 2. 모델 선택
    model_name = st.selectbox("Select GPT Model", ["gpt-5.2",
                                                   "gpt-5-mini",
                                                   "gpt-5-nano",
                                                   #"gpt-5.2-pro",
                                                   "gpt-4o",
                                                   "gpt-4o-mini",                                                   
                                                   "gpt-3.5-turbo"])

    # 3. Instruction 파일 업로드
    instruction_file = st.file_uploader("Upload Instruction Prompt File (.docx)", type=["docx"])
    
    # 4. Output Format 파일 업로드
    output_format_file = st.file_uploader("Upload Output Format Prompt File (.docx)", type=["docx"])
    
    with st.expander("Output format prompt 형식 예시"):
        st.markdown("""
        **작성 예시:**
        ```text
        Greatest dimension: float (tenths of a centimeter)
        Local invasion: string (present vs absent)
        Metastatic lymph node: string (location)
        ```
        """)

    # 5. 데이터 엑셀 파일 업로드
    data_file = st.file_uploader("Upload Data Excel File (.xlsx)", type=["xlsx"])
    
    id_column = None
    input_column = None
    df = None

    if data_file:
        df = pd.read_excel(data_file)
        cols = df.columns.tolist()
        
        # 6. 컬럼 선택
        id_column = st.selectbox("Select ID Column (인덱스용 컬럼)", [""] + cols)
        if id_column:
            st.write(f"미리보기: {df[id_column].iloc[0]}")
            
        input_column = st.selectbox("Select Input Data Column (정보 추출할 텍스트 컬럼)", [""] + cols)
        if input_column:
            st.write(f"미리보기: {str(df[input_column].iloc[0])[:50]}...")

    # ==========================================
    # 3. 실행 로직 (버그 수정됨)
    # ==========================================
    
    if st.button("Submit"):
        if not (api_key and instruction_file and output_format_file and data_file and id_column and input_column):
            st.warning("모든 필드를 입력하고 파일을 업로드해주세요.")
            return

        instruction_text = read_docx(instruction_file)
        format_text = read_docx(output_format_file)
        
        # 스키마 생성
        json_schema = parse_format_to_schema(format_text)
        
        st.info(f"작업을 시작합니다... ({model_name})")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # LLM 초기화
        llm = ChatOpenAI(
            model=model_name, 
            api_key=api_key, 
            temperature=0
        )
        
        # [핵심 수정] LangChain에 스키마 주입
        # LangChain이 내부적으로 이 스키마를 OpenAI Function Calling 포맷으로 변환합니다.
        structured_llm = llm.with_structured_output(json_schema)
        
        results = []
        total_rows = len(df)
        
        for index, row in df.iterrows():
            try:
                input_data = row[input_column]
                
                # 메시지 구성
                messages = [
                    ("system", f"{instruction_text}\n\n당신은 의료 데이터 추출 전문가입니다. 다음 JSON 스키마에 맞춰 정확한 값을 추출하세요."),
                    ("user", f"Report:\n{input_data}")
                ]
                
                # 실행
                response = structured_llm.invoke(messages)
                
                # 결과 처리
                output_row = response if isinstance(response, dict) else response.dict()
                output_row[id_column] = row[id_column]
                results.append(output_row)
                
            except Exception as e:
                # 에러 발생 시 로그 출력
                st.error(f"Error at ID {row[id_column]}: {e}")
                error_row = {id_column: row[id_column], "error_msg": str(e)}
                results.append(error_row)
            
            # 진행률 업데이트
            progress = (index + 1) / total_rows
            progress_bar.progress(progress)
            status_text.text(f"처리 중... {index + 1}/{total_rows}")

        # ==========================================
        # 4. 결과 다운로드
        # ==========================================
        st.success("작업 완료!")
        
        result_df = pd.DataFrame(results)
        
        # 컬럼 순서 정리
        cols = [id_column] + [c for c in result_df.columns if c != id_column]
        result_df = result_df[cols]
        
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            result_df.to_excel(writer, index=False)
        output.seek(0)
        
        filename = f"Extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
        
        st.download_button(
            label="Download Output Excel",
            data=output,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if __name__ == "__main__":
    main()