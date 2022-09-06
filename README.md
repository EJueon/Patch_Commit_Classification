# Patch_Commit_Classification
![ex](./assets/01.png)

## Quick Start
1. git clone https://github.com/EJueon/Patch_Commit_Classification.git
2. 환경 설치 : conda env create -n envname --file environment.yml
3. python main.py를 하면 프로그램이 실행됨
4. [기존 데이터 불러오기] -> ./data/syzbot-data.pickle 또는 ./data/test_data.pickle.gz 선택
5. 상단 상태 창에 '데이터를 불러왔습니다'일 경우, [불러온 데이터 분류하기]버튼을 선택
6. 결과: 데이터에 대한 정확도와 취약점으로 분류한 commit no.이 하단에 표시된다. 
