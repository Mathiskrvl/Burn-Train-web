export function initChart(label, f_wasm_array) {
    const ctx = document.getElementById('myChart').getContext('2d');
    const data = {
        labels: label,
        datasets: [{
            label: 'f(x)',
            backgroundColor: 'rgb(255, 99, 132)',
            borderColor: 'rgb(255, 99, 132)',
            data: f_wasm_array,
            fill: false,
        },
        {
            label: 'g(x)',
            backgroundColor: 'rgb(99, 99, 252)',
            borderColor: 'rgb(99, 99, 252)',
            data: [],
            fill: false,
        }]
    }
    const config = {
        type: 'line',
        data: data,
        options: {}
    };
    return new Chart(ctx, config)
}

// Fonction pour mettre à jour le graphique avec de nouvelles données
export function updateChart(myChart, newData) {
    myChart.data.datasets[1].data = newData;
    myChart.update();
}