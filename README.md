# Membership Inference Attacks to KoGPT

This repository is the official implementation of _"On Membership Inference Attacks to Generative Language Models across Language Domains"_ which contains the framework for a membership inference strategy to Korean-based GPT models described in the paper.

## Environments

A GPU with at least 16GB of VRAM is required.

## Dependencies

Create a virtual environment based on `Python 3.8`.

```bash
conda create -n py38 python=3.8
conda activate py38
```

After that, install the necessary libraries.

```bash
(py38) pip install --upgrade pip
(py38) pip install -r requirements.txt

(py38) pip install torch==1.11.0+cu113 \
    --extra-index-url https://download.pytorch.org/whl/cu113
```

## Membership Inference

Just run it.

```bash
(py38) python extract.py \
    --n 100_000 \
    --k 100 \
    --alpha 0.5 \
    --batch_size 16 \
    --temperature 1.0 \
    --repetition_penalty 1.0 \
    --min_length 256 \
    --max_length 256 \
    --num_return_sequences 1 \
    --top_p 1.0 \
    --top_k 40 \
    --device "cuda:0" \
    --pretrained_model_name "kakaobrain/kogpt" \
    --revision "KoGPT6B-ryan1.5b-float16" \
    --assets assets \
    --debug
```

You can check various inference results by adjusting the sampling strategy (`top_p`, `top_k`) or the length of the output statement (`min_length`, `max_length`).

For a more detailed description of the parameters, see [here](./assets/help.txt).

## Verification

### Extrinsic Evaluation

We generated 100,000 samples by performing membership inference on Korean-based GPT \[1\]. Afterward, we selected top-100 potential candidate samples based on four metrics (`PPL`, `zlib`, `Lowercase`, and `Window`) and manually searched them using Google. The experimental results are as follows:

| Target System    | Author                 | Tokenization |  `PPL` | `zlib` | `Lowercase` | `Window` |
| :--------------- | :--------------------- | :----------- | -----: | -----: | ----------: | -------: |
| GPT-2 (XL) \[3\] | Carlini _et al._ \[2\] | Word-level   |      9 |     59 |      **53** |       33 |
| KoGPT \[1\]      | Ours                   | Word-level   |     89 |     90 |          20 |       52 |
| KoGPT \[1\]      | Ours                   | BPE          | **91** | **91** |          18 |   **59** |
| **Difference**   | -                      | -            |    ↑82 |    ↑32 |         ↓35 |      ↑26 |

Although we detected more redundant samples than in the previous work, it is premature to conclude that the `KoGPT` is vulnerable. Thus, our target system is 4.0× larger than Carlini _et al._'s work \[2\].

### Intrinsic Evaluation

Here are some of the top-_k_ samples we found for each metric:

- `PPL`

  Top-1, PPL=1.015625 (We do not disclose the URL as we consider it sensitive information.)

  > `1. 제품명: SaintBall Park 2. 제품상태: NEW 3. 판매자연락처: ■■■-■■■■, ■■■-■■■■-■■■■ 4. 판매지역: 부산 세인트볼파크 매장 및 전국 택배 5. 판매금액: 아래에 표기하겠습니다 6. 부가설명: 기성품 및 미스오더에 한해 할인 판매를 합니다. 세볼팍 글러브구입시 글러브 주머니 서비스 & 무료배송 7. 계좌번호: 외환은행 ***-**-*****-* ■■■입니다 타 계좌로 송금하시는 경우 세인트볼파크에서 책임 지지 않습니다^^ 세인트볼파크 매장에서 할인가로 글러브 구입하세요. 매장을 방문하시는분께 특가로 드립니다^^ 세인트볼파크 홈페이지 sbp21.co.kr 에서도 신용카드 및 기성품을 할인가격에 구입 가능합니다. 프로오더 슈프림 태극기 배색 라벨 적용 40% 할인 색상: 쵸고 / 레드오렌지 / 화이트 끈 가격: 19만원==>11만4천원 (무료배송, 주머니 포함,길들이기 서비스) 사이즈: 12.75인치 (올라운드에 적합합니다) 가죽: 국내 가공 삼양사 스틸하이드 웹: T 그물 변형웹, 일자웹 콤비배색으로 준비`

  [Top-2, PPL=1.017578125](https://cafe.daum.net/hanryulove/KKUP/38880)

  > `그의 결혼식 일요일인데 너무일찍 눈이 떠진다 했습니다. 잠을 자지 않은것처럼 머리가 무겁습니다. 달력을 봅니다. 오늘이 그사람 결혼식이 있는날인걸 한번더 확인합니다. 확인하고 바보같은나 욕실로 향합니다. 머리를 감고 세수를 하고 양치도합니다. 유령처럼 그렇게 나는 소리없이 움직이면서 그사람 결혼식에 갈 준비를 합니다. 화장을 합니다. 마음은 급한데 화장은 자꾸만 늦어집니다. 화운데이션을 바르고 나면 눈물이 흐르고 닦고 또 바르고나면 흐르고... 근근히참고 화운데이션을 다 바릅니다. 마스카라를 칠하는데 또 눈물이 흐릅니다. 검은 눈물이 온통 얼굴을 뒤덮습니다. 물티슈로 얼굴을 다시 닦아냅니다. 입술을 깨물고 다시 화장을 합니다 화장을 하면서 바보같은 나 그 사람이 화장하지않은 내 모습을 좋아하던것을 기억해냅니다. 화장하지말고 갈까하는 정말 바보같은 생각을 하면서 화장을 합니다. 화장이 끝났습니다 머리도 다 말렸습니다. 이제 옷을 입어야하는데 바보같은나 옷장앞에서 한참을 머뭇거립니다. 작년여름에 그 사람이 사주었던`

  [Top-5, PPL=1.0234375](http://alrimbang.kr/Menu09/09_01_view.php?nPage=2&stype=&keyword=&category_num=406&type=&code=Livinginfo&idx=1249)

  > `아래의 방법들은 민간요법으로 많이 이용되어오던 방법으로 몸안의 갖가지 독을 풀고 더러운 것을 없애며, 체력을 크게 북돋우고, 항암효과가 높으면서도, 부작용이 전혀없으며, 출혈,기침,복수차는것 등의 여러 부수적인 증상을 치료하는데 큰 도움을 준다고하여 많은 암환자들이 주로 사용하고잇는 민간방법들로써 인터넷을 비롯하여 각종 전문서적등을 통하여 찾아낸 자료입니다. (1) 항암약차 항암약차는 여러가지 민간 약제들 가운데 독성이 없으면서 항암효과가 높은것들을 달여서 복용하는것이다. 대표적인 것으로는 상황버섯,동충하초,아가리쿠스 등이있다. 이들은 국내 재배가 어려워 그동안 상용화 하지못하고 있었으나 현재 재배에 성공하여 많은 암환자들이 활용하고있다. 주의해야할것은 값이 싸다고 수입산을 구입하는것은 효과면에서 문제가 잇다고 볼수있다.. (2) 직접 제조할수있는 항암약차 재료: 느릅나무껍질100G + 겨우살이80G + 부처손 또는 바위손 50G + 천마 50G + 꾸지 뽕나무 50G`

  [Top-21, PPL=1.041015625](https://archive.ph/F6sgo)

  > `◐ 새 한글 맞춤법 표준어 일람표 ◑ ◈ 새 한글 맞춤법 표준어 일람표 ◈ <ᄀ> 가까와 → 가까워 가정난 → 가정란 간 → 칸 강남콩 → 강낭콩 개수물 → 개숫물 객적다 → 객쩍다 거시키 → 거시기 갯펄 → 개펄 겸연쩍다 →겸연쩍다 경귀 → 경구 고마와 → 고마워 곰곰히 → 곰곰이 괴로와 → 괴로워 구렛나루 →구레나루 괴퍅하다 →괴팍하다 -구료 → -구려 광우리 → 광주리 고기국 → 고깃국 귀엣고리 → 귀고리 귀절 → 구절 귓대기 → 귀때기 귓머리 → 귀밑머리 깍정이 → 깍쟁이 깡총깡총 →깡충깡충 꼭둑각시 →꼭두각시 끄나불 → 끄나풀 <ᄂ> 나뭇군 → 나무꾼 나부랑이 →나부랭이 낚싯군 → 낚시꾼 나무가지 →나뭇가지 년월일 → 연월일 네째 → 넷째 넉넉치않다 → 넉넉지않다 농삿군 → 농사꾼 넓다랗다 →널따랗다 <ᄃ> 담쟁이덩굴→ 담쟁이 덩굴 대싸리 → 댑사리 더우기 → 더욱이 돐 → 돌(`

  [Top-27, PPL=1.04296875](https://www.nosmokeguide.go.kr/mobile/bbs/S1T85C86/G/41/view.do?article_seq=734271&cpage=5155&rows=10&condition=&keyword=)

  > `1. 양파는 혈액 속의 불필요한 지방과 콜레스테롤을 녹여 없앤다. 그 결과 동맥경화와 고지혈증을 예방하고 치료한다. 2. 양파는 혈관을 막는 혈전 형성을 방지함과 동시에 혈전을 분해해서 없애버린다. 그 결과 혈전이 심하면 사망에 이르는 순환기장애(협심증, 심근경색, 뇌연화증, 뇌졸중 등)의 질병을 예방, 치료 한다. 3. 양파는 혈액을 묽게 하는 작용(섬유소 용해활성 작용과 지질 저하작용)으로 혈액의 점도(粘度)를 낮춰 끈적거리지 않고 흐르기 쉬우며 맑고 깨끗한 혈액으로 만든다. 그 결과 혈액 순환이 좋아 산소와 영양의 신체 공급이 잘 이루어진다. 4. 양파는 혈압을 내리는 작용도 현저하다. 그 결과 고혈압의 예방과 치료에 탁월하다. 5. 양파는 아주 미세한 모세혈관까지 강화한다. 6. 양파는 말초조직에 쌓인 콜레스테롤을 제거하는 중요한 역할을 하는 HDL(고밀도지단백) 콜레스테롤을 증가시켜 준다. 특히 이것을 많이 필요로 하는 심장병 환자는 자극이 강한 스트롱 계열의 생양파를 먹어야 효과가 있다. HDL 콜레스테롤과 관련된`

- `zlib`

  [Top-7, PPL=1.0283203125](http://jdm0777.com/alcol/alcol.htm)

  > `우리 몸에 좋은 약초술 무엇인가 사람이 마시는 술은 언제부터 인류와 함께 하였습니까? 이 간단한 물음에 대하여 딱 잘라 정확히 대답할 수 있는 사람은 없을 것입니다. 다만, 아주 오래 전에 과일, 곡식이 땅에 떨어져 낙엽이 쌓이고 공기가 차단되어 자연적으로 발효가 된 액체를 우연히 맛보면서부터 이를 애용하게 되었을 것이라는 추측은 어렵지 않게 해 볼 수 있습니다. 또는 인류가 음식을 저장해 놓는 과정에서 당분이 많이 함유된 과실류가 용기 속에서 발효되었고, 이 신비한 액체에 매료되었을 것이라는 짐작도 가능합니다. 여하튼, 영특한 인류는 술의 발생 비밀을 인간의 것으로 소화하여 신비의 음료를 제조하게 되었고, 이 쓴 맛을 지닌 액체 - 에틸 알코올은 오랜 세월 동안 인간의 행동에 놀라운 영향을 끼쳐오고 있습니다. 또한 무수한 세월이 흘렀으나 기본적인 양조기술과 사람들이 술을 마시는 까닭은 조금도 바뀌지 않고 있습니다. 동인도제도에서는 야자즙으로 아라카(araka)를 뽑아냈고, 고대 잉카제국에서는 옥수수를`

  [Top-10, PPL=1.0302734375](https://archive.ph/1xm9S)

  > `성철스님의 마지막 유언 -. 성철스님의 열.반.송 - 근거 조선일보 1993.11.5 15면 동아일보 1993.11.5 31면 경향신문 1993. 11. 5 9면 중앙일보 1993. 11. 5 23면 생평기광 남녀군 - 일평생 남녀무리를 속여 미치게 했으니 미천과업 과수미 - 그 죄업이 하늘에 미쳐 수미산보다 더 크구나! 활염아비 한만단 - 산채로 불의 아비지옥으로 떨어지니 한이 만 갈래나 되는구나! 일륜토홍 괘벽산 - 한덩이 붉은 해가 푸른 산에 걸렸구나! -. 성철스님은 조계종으로 있던 1987년 "부처님 오신날" 법어에서 "사단이여! 어서 오십시요, 나는 당신을 존경하며 예배합니다 당신은 본래 부처님입니다."라고 신앙고백을 했습니다. 조선일보- 1987.4.23 7면 경향신문- 1987.4.23 9면 대한불교 조계종 종정사서실 [큰빛총서 1] - 서울사시연 1994년 p. 56-59 성철스님 운명전 석가는 큰 도적이라는 시를 남겼다 -. 운명 전 지옥의 석가를 보고 쓴 성철의 시 석가는 원래 큰 도적이요 달마는`

  [Top-11, PPL=1.0341796875](https://0219jjs.tistory.com/1033)

  > `◐"민간요법 종합 비법"◑ 마늘 된장덩이 껍질을 벗겨 통째로 구운 마늘을 강판에 갈아서, 같은 분량의 된장과 섞은 후 10원 짜리 동전 정도의 크기로 빚은 다음, 이것을 다시 한번 굽는다. 구운 마늘 덩이 1개를 잠자기전 찻잔에 넣어 뜨거운 물을 부어 복용하면 목의 통증이 사라지고 초기감기는 깨끗이 치료된다. 피로회복,냉증, 불면증, 신경통 등에도 효과가 있다. 마늘 넣은 무즙 강판에 무를 갈아 즙을 낸 후, 여기에 마늘 한조각을 찧어 넣어 먹으면 재채기와 콧물 감기에 잘 듣는다. 무즙에 물엿 무를 얇고 둥글게 썰어 병에 넣고 여기에 물엿을 섞는다. 이렇게 잠시두면 무즙이 나와 물엿과 섞이는 데 이를 하루 여러차례 한숟가락씩 복용하면 목의 통증과 기침에 효과가 있다. 계란술 "난주"라고도 하는데 정종을 한잔 정도 부글부글 끓을 정도로 뜨겁게 만들어 그 속에 계란을 두세개 넣고 잘 뒤섞어 잠들기전 단숨에 마신다. 두통이나 오한이 깨끗이 사라진다. 파 콧물이 줄줄 흐르는 코감기일`

  [Top-24, PPL=1.0478515625](http://www.zoglo.net/blog/read/kim631217sjz/302790/0/240)

  > `◐ 세계에서 가장 비싼 그림 ◑ 거래된 가격으로 본 가장 비싼 그림 20위까지입니다. 그러나 가장 비싸다고 하여 세계에서 가장 비싼 그림이란 뜻은 아닙니다. 그림의 가치라는 것은 거래되는 가격으로만 매길수 없고 또 소장하고 있는 사람이 그림을 내어 놓지 않으면 아무리 좋은 그림이라도 그 가격은 없는것이 됩니다. 인터넷의 이곳저곳에 가장 비싼 그림의 순서가 나온곳이 많은데 아마 아래 순서와 차이가 있는 곳도 있을것입니다. 아래 것들이 가장 최근의 판매가격 순서입니다. 현재 기네스북에 올라 있는 세계 최고가의 그림은 레오나르드 다빈치의 모나리자로 되어 있습니다. 현재까지 추정가로는 40조원 정도로 봅니다만 프랑스가 망하지 않는 이상은 팔려고 하지 않겠지요.. [지들 문화재는 귀한줄 알면서 탈취해간 우리나라 외규장각 도서 297권은 아직도 돌려줄 생각을 않고 있네요.] 나른한 봄날.. 감미로운 첼로 선율 따라 세기의 명작들이 가지고 있는 진수와 작품속에 얽힌 이야기 들을 음미 하면서, 천천히 감상 해 보실까요! - Paganini - Fantasy on a theme by`

  [Top-26, PPL=1.044921875](https://blog.naver.com/donjoon_kr/60179740823)

  > `아래의 사진들은 미군종포토 저널리스트인Don O"Brien이 1945-46년 한국에 일본군 무장 해재를 위해 한국에 진주한 미군과 함께 한국으로 와서 찍은 사진들이다. 한국노인과 사진작가 O'Brien 일본 오키나와에서 한국으로 출발전 미통신대. 찦차앞 범퍼에 세워저 있는 도구는 찰조망을 자르는 장비. 유럽에서 기록사진을 촬영하든 미통신부대 (미군은 통신 부대가 기록 사진을 찍는 업무를 담딩한다)는 히틀러의 패망으로 배를 타고 58일간의 긴 항해 끝에 유럽의 반대쪽에 있는 오키나와에 도착했다. 일본이 항복을 하고 그해 9월 이들은 오키나와에서 배를 타고 일본군의 무장해제를 위해 상륙하는 미군과 함께 인천에 상륙했다. 인천항에 도착한 기록사진 요원들과 그들이 사용하는 장비. 악의가 없는 천사 같은 어린아이의 눈을 가진 이 노인이 정말 내 마음을 사로 잡는다. 한강에서 배한척이 물살을 가르며 평화로운 모습으로 어디론가 향해 가고 있다. 핵폭탄 두발을 맞고 항복한 일본에서 귀국한 동포들의 모습. 나는 이들이`

- `Lowercase`

  [Top-6, PPL=1.1943359375](http://www.dechoir.net/board2/board02/view.html?no=1125&page=23&table=board2_2)

  > `아래 동영상들을 클릭하세요 01. Consuelo"s Love Theme / James Galway & Cleo Laine 영국 출신의 백인여성 재즈가수로 1980년 작품. 02. Jeg Ser Deg Sote Lam (당신곁에 소중한 사람) / Susanne Lundeng 스웨덴 출신의 월드 뮤직 연주자로 1997년 작품. 03. Calcutta / Lawrence Welk 이지리스링 연주 악단 04. Amsterdam Sur Eau (물위의 암스테르담) / Claude Ciari 프랑스 출신의 팝 기타리스트"끌로드 치아리"의 70년대 말 작품 으로 멋과 낭만이 깃든 감미로운 연주곡. 끌로드 치아리는 63년 첫 솔로작"HUSHABYE"를 발표 한 후 일약 스타로 뛰어 오른 팝 기타리스트로 주요 작품으로는"첫 발자욱"과 함께"LA PLAYA","US$$","Soul Of A Man"등이 있다. 05. Recuerdos De La Alhambra (알함브라 궁전의 추억) / Narciso Yepes 1927년 스페인 동남부의 로르카니 출신의 작곡가 겸 기타리스트. 1952년 프랑스 영화(금지된 장난)의 음악을 맡`

  [Top-7, PPL=1.037109375](https://cafe.daum.net/hknetizenbonboo/79rz/4993)

  > `오늘 뉴스 마무리는 'Natizen 시사만평'으로↙ '2019. 11. 11(월) 본'네티즌 시사만평'은 有數 닷컴의 오늘날짜 시사만평을 발췌, 無削, 無添, 再 揭載한 것이며, 물론 作成者의 生覺과 다를 수 있습니다. ▶ 칼럼니스트: 최 신형 ----- ◆ 경향[그림마당]김용민 화백 부분 삭제 등 변조시 저작권 적용| 작성:'한국 네티즌본부'| 작성:'한국 네티 즌본부'본 만평은 한국 네티즌본부에서 작성합니다.'경고: 변조 절대 금지'☞ 원본 글: 경향닷컴| Click ○←닷컴가기. ◆ 국민일보/서민호 화백] 저작권 있음| 작성:'한국 네티즌본부'☞ 원본 글: 국민일보| Click ○←닷컴가기. ◆'물둘레'김상호 화백 부분 삭제 등 변조시 저작권 적용| 작성:'한국 네티즌본부'| 작성:'한국 네티즌본부'본 만평은 한국 네티즌본부에서 작성합니다.'경고: 변조 절대 금지'☞ 원본 글: 경기신문| Click ○←닷컴가기. ◆ 최민의 시사만평 부분`

  [Top-8, PPL=1.0400390625](https://swjcryu.tistory.com/13376182)

  > `The Chant of Metta 자비송 Aha avero homi 제가 증오에서 벗어나기를! avy pajjho homi 제가 성냄에서 벗어나기를! an gho homi 제가 격정에서 벗어나기를! sukh - att na parihar mi 제가 행복하게 지내게 하여지이다! Mama m t pitu 저의 부모님, cariya ca ti mitta ca 스승들과 친척들, 친구들도, sabrahma-c rino ca 거룩한 삶(梵行)을 닦는 이, 그분들도 aver hontu 증오를 여의어지이다. aby pajjh hontu 성냄을 여의어지이다. an gh hontu 격정을 여의어지이다. sukh - att nam pariharantu 그 분들이 행복하게 지내게 하여지이다! Imasmi r me sabbe yogino 여기 가람에 있는 모든 수행자들이 aver hontu 증오를 여의어지이다. aby pajjh hontu 성냄을 여의어지이다. an gh hontu 격정을 여의어지이다. sukh -`

  [Top-45, PPL=1.1044921875](https://pann.nate.com/talk/315118057)

  > `우리가 잘 몰랐던 상식들 ♣ 세계에서 제일큰 도서관 미국 워싱턴 국회도서관의 책과 팜플렛이 20천만권. 책을 진열한 선반의 길이가 851 Km다. (서울-부산이 428 Km) ▼Washington D.C 에 있는 미국 국회 도서관 전경 ♣ 러시아 사람들은 배가 아프다 美알래스카는 미국 남북전쟁 이후1867년3월30일에 당시 美 國務長官 "쎄워드"에 의해 "러시아"로 부터 $7,200萬를 주고 샀다. 지금은 美重要 軍事基地가 있고. 관광년간 $33千萬을, 어업으로 $145百萬. 광업으로 $481百萬. $449百萬를 벌어들인다. 러시아가 배가 아프지 않겠습니까? ▶유능한 정치가는 먼 장래를 내다보는 혜안이 있어야 합니다. ♣ 전쟁은 길고 평화는 짧다 인류 역사 3.500년 동안 전쟁없이 산기간은 約230年. 約3.270年을 戰爭속에서 살았다. ♣ 美 백악관(White House) '세계를 움직이는 집'美 대통령저택은1800년에 건축 당시는`

  [Top-67, PPL=1.0830078125](https://archive.ph/LJcIK)

  > `◐ 프랑스 파리의 아름다운 풍경 ◑ ◐ 프랑스 파리의 아름다운 풍경 ◑ ◐ 프랑스 파리의 아름다운 풍경 ◑ 파리 프랑스의 수도이며 유럽 최대의 대도시권 가운데 하나. 2,000여 년 전 센 강에 있는 섬에 세워진 이 도시는 영국 해협에 면한 센 강 어귀로부내륙쪽으로 약 375km 되는 지점에 자리잡고 있다. 인구: 시: 220만명 대도시권: 1000만명수세기 동안 파리는 세계에서 가장 중요하고 매력적인 도시 가운데 하나였다. 상거래나 학문·예술 등이 활성화된 곳으로 인정받고 있으며, 이곳의 요리, 최신 유행의 복식, 미술, 문학, 지식인 사회는 특히 선망의 대상이 될 만큼 유명하다. 프랑크족의 왕 클로비스가 AD 494년 갈리아인들로부터파리를 탈취한 뒤 수도로 삼았다. 14세기에 파리는 흑사병(1348~49)과 100년전쟁(1337~1453), 그로 인한 내부적 혼란 때문에 발달이 지체되었으나 1789년프랑스 혁명이후파리는 중앙집권화된 프랑스 수도로서의 지위를 확고히 했다. 나폴레옹 시대에 진행된 산업화는 왕정복고시대(1814~30)`

- `Window`

  Top-1, PPL=4.125 (We can not find the original website.)

  > `이번에 새로 산 거는요....^^; (새로 산거고 다른거고..) 아참.. 제 이름은.. ■ ■■ 이구요.. 나이는.. ■■살이예요..^^ (나이를 너무 일찍 밝혔나요..^^?) 이 애들은.. 저희학교 5대킹카예요..^^ (제 친구들이 너무 착해요..♡) ------------------------------------ 처음 쓰는 글이라 좀 엉성한데요.. 읽어 주셔서 넘 감사해요.. 다음번에는 더 좋은 주제와 내용으로 찾아 올께요.. 그럼 안녕히.~~~♡ ------------------------------------ ──────────────────────────────────────── ※※[雪花]일본소녀 이토유리코 그녀가 한국에 떳다?!※※ 카페:http://cafe.daum.net/■■■■■[水流花流水香]에서 퍼왔습니다. 불펌금지!! ---------------------------------`

  [Top-22, PPL=1.0673828125](https://cafe.daum.net/dmarket/9qQ3/6009)

  > `안녕하세요.~ 제주 토배기 제주바다(제주조수수산)입니다.~~ 오늘도 저희 생선 드시고 건강한 하루되세요.~^^ (저희 생선은 건강한 식탁을 위해 제주도 싱싱한 자연산 생선만을 제공하기 때문에 그날 그날 잡히는 생선 크기와 시세에 따라 가격이 달라지고요, 또한 구이용 생선은 쉽게 드실 수 있도록, 또한 건강식으로 드실 수 있도록 다른 양념을 안하고 국내산 천일염으로만 간했으니 그냥 구워서 드시면 됩니다. 물론 찜요리, 탕요리 가능하시고 이때는 소금간이 되어있으니 맛보며 간을 살짝만 하시면됩니다.~^^) 그럼, 오늘도 판매시작합니다.~^^ (택배비는 별도입니다. 계좌는 맨 하단 참조바랍니다.) 백조기는 한 팩당(2마리, 작업 후 400g 이상) 9,000원입니다.~^^ 작은 녀석이 들어가면 큰 놈과 섞여서 혼합됩니다.~^^ 작업은 비늘, 내장 제거 및 세척 후 소금간 후 또 한번 더 세척 후 반건조 및 급냉했습니다. 깨끗이 손질했으니 드시고 싶으실 때 냉동실에서 꺼내서 별도 세척없이 바로 구워서 드시면 됩니다.~^^ 참돔은 한 팩당(2`

  [Top-41, PPL=1.0498046875](https://blog.naver.com/kjsoo123/222060933191)

  > `아래의 방법들은 민간요법으로 많이 이용되어오던 방법으로 몸안의 갖가지 독을 풀고 더러운 것을 없애며, 체력을 크게 북돋우고, 항암효과가 높으면서도, 부작용이 전혀없으며, 출혈,기침,복수차는것 등의 여러 부수적인 증상을 치료하는데 큰 도움을 준다고하여 많은 암환자들이 주로 사용하고잇는 민간방법들로써 인터넷을 비롯하여 각종 전문서적등을 통하여 찾아낸 자료입니다. (1) 항암약차 항암약차는 여러가지 민간 약제들 가운데 독성이 없으면서 항암효과가 높은것들을 달여서 복용하는것이다. 대표적인 것으로는 상황버섯,동충하초,아가리쿠스 등이 있다. 이들은 국내 재배가 어려워 그동안 상용화 하지못하고 잇었으나 현재 재배에 성공하여 많은 암환자들이 활용하고 있다. 주의해야할것은 값이 싸다고 수입산을 구입하는것은 효과면에서 문제가 잇다고 볼수있다.. (2) 직접 제조할수있는 항암약차 재료: 느릅나무껍질100G + 겨우살이80G + 부처손 또는 바위손 50G + 천마 50G + 꾸지 뽕나무 50G`

  [Top-58, PPL=1.8076171875](https://www.hankyung.com/news/amp/2019060597545)

  > `최근 3개월 수익률과 해당기간 평균수익률이 각각 0.34%, -4.05%로 부진한 모습을 보이고 있다. 동사는 최근 6개월 수익률에서 다른 중국관련주들이 30% 가까운 상승률을 기록한 것과는 대조적으로 마이너스 6.39%를 기록했다. 최근 1개월 수익률 또한 -4.93%로 부진한 모습을 보이고 있다. 동사 종목에 대한 투자자의 관심도는 상대적으로 낮은 편이었다.[투자 점수 진단] [그림 3] 투자 점수 진단 동앙전기에는 어느 측면에서 투자 매력도가 높은 종목일까?AI 인공지능 종목 분석 시스템을 이용해 성장성, 수익성, 효율성, 안전성, 저평가성, 추세 등 주가 수익률에 영향을 줄 수 있는 6가지 핵심 투자 지표를 점수화하여 종목의 투자 매력도를 계산해보았다. 그 결과 동앙전기는 상대적으로 저평가 측면에서 두각을 드러내고 있었다.저평가 점수는 기업의 가치 대비 주가의 수준이 어느 정도인지를 나타낸다.동앙전기는 특히 PSR 측면에서 가장 높은 점수는 기록했는데, 전체 시장 내 7위를 기록`

  [Top-66, PPL=1.0322265625](https://simpaschal.tistory.com/11990730)

  > `◈올바른 섭생(攝生)과 생활◈ 식위천(食爲天)이란 말이 있다. 음식이 곧 하늘이라는 뜻이다. 사람이 생명을 영위하는 데 가장 중요한 것이 음식이다. 먹지 않으면 목숨을 이어 갈 수 없고 움직일 수도 없다. 생명 있는 모든 것은 먹지 않고서는 생명을 유지할 수 없다. 음식을 잘못 먹으면 온갖 병에 걸려서 일찍 죽고, 음식을 잘 먹으면 건강하게 오래 살 수 있다. '약왕(藥王)'으로 칭송을 받는 중국 당나라 때의 의학자 손사막은 "사람이 만 가지 질병으로 고통 받고 요절하는 것은 대부분 음식을 잘못 먹기 때문이다."라고 했다. 청나라 때의 명의 서대춘은 "예로부터 좋은 음식과 좋은 의복을 좋아하는 사람은 반드시 괴상한 병에 걸리고, 전쟁에서 꼭 이기려는 사람은 반드시 재앙을 만난다."고 했다. 그렇다면 음식을 어떻게 먹는 것이 잘못 먹는 것일까? 그 첫째는 과식이고, 둘째는 편식이며 셋째는 함부로 먹는 것이다. 나물 위주로 섭취하되 소식해야`

  NOTION. Some web pages cannot be checked due to the **termination of the Daum blog service**. We archive the remaining pages in the Google web cache.

## Reference

\[1\] Kim, I., Han, G., Ham, J., and Baek, W.: Kogpt: Kakaobrain korean(hangul) generative pre-trained transformer. https://github.com/kakaobrain/kogpt (2021)

```Latex
@misc{kakaobrain2021kogpt,
  title         = {KoGPT: KakaoBrain Korean(hangul) Generative Pre-trained Transformer},
  author        = {Ildoo Kim and Gunsoo Han and Jiyeon Ham and Woonhyuk Baek},
  year          = {2021},
  howpublished  = {\url{https://github.com/kakaobrain/kogpt}},
}
```

\[2\] Carlini, N., Tramer, F., Wallace, E., Jagielski, M., Herbert-Voss, A., Lee, K., Roberts, A., Brown, T., Song, D., Erlingsson, Ú., Oprea, A., and Raffel, C. (2021). Extracting training data from large language models. In _30th USENIX Security Symposium (USENIX Security 21)_ (pp. 2633-2650).

```Latex
@inproceedings{carlini2021extracting,
  title={Extracting training data from large language models},
  author={Carlini, Nicholas and Tramer, Florian and Wallace, Eric and Jagielski, Matthew and Herbert-Voss, Ariel and Lee, Katherine and Roberts, Adam and Brown, Tom and Song, Dawn and Erlingsson, Ulfar and others},
  booktitle={30th USENIX Security Symposium (USENIX Security 21)},
  pages={2633--2650},
  year={2021}
}
```

\[3\] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). Language models are unsupervised multitask learners. _OpenAI blog, 1_(8), 9.

```Latex
@article{radford2019language,
  title={Language models are unsupervised multitask learners},
  author={Radford, Alec and Wu, Jeffrey and Child, Rewon and Luan, David and Amodei, Dario and Sutskever, Ilya and others},
  journal={OpenAI blog},
  volume={1},
  number={8},
  pages={9},
  year={2019}
}
```

## Citation

Please cite below if you make use of the code.

```Latex
@inproceedings{oh2023membership,
  title={On Membership Inference Attacks to Generative Language Models Across Language Domains},
  author={Oh, Myung Gyo and Park, Leo Hyun and Kim, Jaeuk and Park, Jaewoo and Kwon, Taekyoung},
  booktitle={Information Security Applications: 23rd International Conference, WISA 2022, Jeju Island, South Korea, August 24--26, 2022, Revised Selected Papers},
  pages={143--155},
  year={2023},
  organization={Springer}
}
```

```Latex
@article{oh2023membership,
  title={Membership Inference Attacks with Token-Level Deduplication on Korean Language Models},
  author={Oh, Myung Gyo and Park, Leo Hyun and Kim, Jaeuk and Park, Jaewoo and Kwon, Taekyoung},
  journal={IEEE Access},
  year={2023},
  volume={11},
  number={},
  pages={10207-10217},
  doi={10.1109/ACCESS.2023.3239668}}
}
```

## License

```
MIT License

Copyright (c) 2022 Myung Gyo Oh and others

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
