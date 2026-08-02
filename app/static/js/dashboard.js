const growth = document.getElementById("growth-values");

const waz = Number(growth.dataset.waz);
const haz = Number(growth.dataset.haz);
const whz = Number(growth.dataset.whz);

const ctx = document.getElementById("growthChart");

new Chart(ctx, {

    type: "bar",

    data: {

        labels: ["WAZ", "HAZ", "WHZ"],

        datasets: [{

            label: "WHO Z-Scores",

            data: [waz, haz, whz],

            backgroundColor: [
                "#4CAF50",
                "#2196F3",
                "#FF9800"
            ],

            borderWidth: 1

        }]
    },

    options: {

        responsive: true,

        scales: {

            y: {

                min: -5,

                max: 5

            }
        }
    }
});