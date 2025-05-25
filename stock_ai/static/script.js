document.addEventListener('DOMContentLoaded', function () {
    const ctx = document.getElementById('priceChart').getContext('2d');
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('error-message'); // ID 일치
    const stockSelector = document.getElementById('stockSelector');
    const stockSymbolTitle = document.getElementById('stockSymbolTitle');
    
    const addDataForm = document.getElementById('addDataForm');
    const addPredictionResultDiv = document.getElementById('addPredictionResult');

    const tradeDateInput = document.getElementById('tradeDate');
    const openPriceInput = document.getElementById('openPrice');
    const highPriceInput = document.getElementById('highPrice');
    const lowPriceInput = document.getElementById('lowPrice');
    const closePriceInput = document.getElementById('closePrice');
    const volumeInput = document.getElementById('volume');
    const adjClosePriceInput = document.getElementById('adjClosePrice');
    const marketCapInput = document.getElementById('marketCap');

    const volumeInputDiv = document.getElementById('volumeInputDiv');
    const adjCloseInputDiv = document.getElementById('adjCloseInputDiv');
    const marketCapInputDiv = document.getElementById('marketCapInputDiv');

    let priceChart; 

    function showLoading(show) {
        loadingDiv.style.display = show ? 'block' : 'none';
    }

    function showError(message) {
        errorDiv.textContent = message;
        errorDiv.style.display = message ? 'block' : 'none';
    }
    
    function toggleExtraInputFields(symbol) {
        // 기본적으로 모두 숨김
        adjCloseInputDiv.style.display = 'none';
        marketCapInputDiv.style.display = 'none';
        volumeInputDiv.style.display = 'block'; // Volume은 대부분 있음

        if (symbol.startsWith("ETH")) {
            marketCapInputDiv.style.display = 'flex'; // flex로 변경
            volumeInput.placeholder = "예: 369,583,008 (쉼표 포함 가능)";
        } else if (symbol.startsWith("SAMSUNG")) {
            adjCloseInputDiv.style.display = 'flex'; // flex로 변경
            volumeInput.placeholder = "예: 10588400 (숫자만)";
        }
    }


    function updateChart(chartData) {
        const labels = chartData.dates;
        const actualPrices = chartData.prices.map((price, index) => ({x: labels[index], y: price}));

        const datasets = [
            {
                label: `${chartData.symbol} 실제 중간 가격`,
                data: actualPrices,
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1,
                fill: false
            }
        ];

        if (chartData.initial_prediction) {
            const lastActualLabel = labels[labels.length - 1];
            const initialPredLabel = chartData.initial_prediction.date;
            const lastActualPrice = chartData.prices[chartData.prices.length - 1]; // prices는 중간 가격 배열
            const initialPredPrice = chartData.initial_prediction.price;

            datasets.push({
                label: '초기 다음 날 예측',
                data: [
                    { x: lastActualLabel, y: lastActualPrice },
                    { x: initialPredLabel, y: initialPredPrice }
                ],
                borderColor: 'rgb(255, 99, 132)',
                borderDash: [5, 5],
                tension: 0.1,
                fill: false,
                pointRadius: 6,
                pointBackgroundColor: 'rgb(255, 99, 132)'
            });
        }
        
        if (chartData.new_prediction_data) { // add_and_predict의 응답을 new_prediction_data로 받음
            const userInputDate = chartData.new_prediction_data.user_input_date;
            const userInputMidPrice = chartData.new_prediction_data.user_input_mid_price;
            const predictedDate = chartData.new_prediction_data.predicted_date;
            const predictedPrice = chartData.new_prediction_data.predicted_price;

            // 사용자가 입력한 데이터 포인트 (차트에서 실제 데이터처럼 보이도록)
            datasets[0].data.push({x: userInputDate, y: userInputMidPrice }); 
            // 날짜 레이블에도 추가 (중복 방지 및 정렬 필요할 수 있음)
            if (!labels.includes(userInputDate)) {
                labels.push(userInputDate);
            }


            datasets.push({
                label: '사용자 입력 기반 예측',
                data: [
                    { x: userInputDate, y: userInputMidPrice },
                    { x: predictedDate, y: predictedPrice }
                ],
                borderColor: 'rgb(153, 102, 255)',
                borderDash: [5, 5],
                tension: 0.1,
                fill: false,
                pointRadius: 6,
                pointBackgroundColor: 'rgb(153, 102, 255)'
            });
        }

        // 날짜 레이블 정렬 (중복 제거 및 시간순)
        const allDates = Array.from(new Set(labels
            .concat(chartData.initial_prediction ? [chartData.initial_prediction.date] : [])
            .concat(chartData.new_prediction_data ? [chartData.new_prediction_data.user_input_date, chartData.new_prediction_data.predicted_date] : [])
        )).sort();


        if (priceChart) {
            priceChart.destroy();
        }

        priceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: allDates, 
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'day',
                            tooltipFormat: 'yyyy-MM-dd', // date-fns 형식
                            displayFormats: {
                                day: 'yyyy-MM-dd'
                            }
                        },
                        title: { display: true, text: '날짜' }
                    },
                    y: { 
                        title: { display: true, text: '가격' },
                        beginAtZero: false // 가격이 0부터 시작하지 않도록
                    }
                },
                plugins: {
                    tooltip: { mode: 'index', intersect: false },
                    legend: { position: 'top' }
                }
            }
        });
    }

    async function fetchStockData(symbol) {
        showLoading(true);
        showError('');
        addPredictionResultDiv.innerHTML = '';
        stockSymbolTitle.textContent = symbol.split('_')[0]; // 예: ETH, SAMSUNG
        toggleExtraInputFields(symbol);

        try {
            const response = await fetch(`/get_stock_data?symbol=${symbol}`);
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `서버 오류: ${response.status}`);
            }
            const data = await response.json();
            if (data.error) {
                showError(data.error);
            } else {
                const chartDisplayData = {
                    symbol: data.symbol,
                    dates: data.dates,
                    prices: data.prices,
                    initial_prediction: data.initial_prediction
                };
                updateChart(chartDisplayData);
                
                if (data.initial_prediction) {
                    addPredictionResultDiv.innerHTML = `초기 ${data.initial_prediction.date} 예측 가격: <strong>${data.initial_prediction.price.toFixed(2)}</strong>`;
                } else {
                    addPredictionResultDiv.innerHTML = '초기 예측을 수행할 수 없습니다 (모델 또는 데이터 부족).';
                }
            }
        } catch (error) {
            showError(`데이터 로드 실패: ${error.message}`);
            console.error('Fetch error:', error);
        } finally {
            showLoading(false);
        }
    }

    stockSelector.addEventListener('change', function() {
        fetchStockData(this.value);
    });

    addDataForm.addEventListener('submit', async function(event) {
        event.preventDefault();
        showLoading(true);
        showError('');
        addPredictionResultDiv.innerHTML = '';

        const symbol = stockSelector.value;
        
        const payload = {
            symbol: symbol,
            trade_date: tradeDateInput.value,
            open_price: openPriceInput.value,
            high_price: highPriceInput.value,
            low_price: lowPriceInput.value,
            close_price: closePriceInput.value,
            volume: volumeInput.value
        };

        if (symbol.startsWith("ETH")) {
            payload.market_cap = marketCapInput.value;
        } else if (symbol.startsWith("SAMSUNG")) {
            payload.adj_close_price = adjClosePriceInput.value;
        }

        if (!payload.trade_date || !payload.open_price || !payload.high_price || !payload.low_price || !payload.close_price) {
            showError('날짜와 주요 가격 정보는 필수 입력입니다.');
            showLoading(false);
            return;
        }
        if (parseFloat(payload.low_price) > parseFloat(payload.high_price)){
            showError('저가는 고가보다 클 수 없습니다.');
            showLoading(false);
            return;
        }

        try {
            const response = await fetch('/add_and_predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || `서버 오류: ${response.status}`);
            }
            const result = await response.json();

            if (result.error) {
                showError(result.error);
            } else {
                addPredictionResultDiv.innerHTML = `<strong>${result.message}</strong><br>
                    사용자가 입력한 ${result.user_input_date}의 데이터 (중간값: ${result.user_input_mid_price.toFixed(2)})를 기반으로, <br>
                    <strong>${result.predicted_date}의 예측 가격: ${result.predicted_price.toFixed(2)}</strong>`;
                
                // 차트 업데이트 (새로운 데이터와 예측 결과로)
                const chartDisplayData = {
                    symbol: symbol,
                    dates: result.updated_dates,
                    prices: result.updated_prices,
                    initial_prediction: null, // 초기 예측은 더 이상 표시 안 함
                    new_prediction_data: result // add_and_predict의 전체 응답 전달
                };
                updateChart(chartDisplayData);
            }
        } catch (error) {
            showError(`데이터 추가 및 예측 실패: ${error.message}`);
            console.error('Add and Predict error:', error);
        } finally {
            showLoading(false);
        }
    });

    fetchStockData(stockSelector.value); // 페이지 로드 시 기본 심볼 데이터 가져오기
});