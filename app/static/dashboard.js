// ==========================================
// READ DATA FROM FLASK
// ==========================================

function readJSON(id, defaultValue) {

    try {

        const element = document.getElementById(id);

        if (!element) {
            console.error("Missing data element:", id);
            return defaultValue;
        }

        return JSON.parse(element.textContent);

    } catch (error) {

        console.error("Chart data error:", error);

        return defaultValue;
    }
}


// Get dashboard data

const normalCount = Number(
    readJSON("normal-count-data", 0)
);

const anomalyCount = Number(
    readJSON("anomaly-count-data", 0)
);

const attackLabels = readJSON(
    "attack-labels", []
);

const attackCounts = readJSON(
    "attack-counts", []
);

const clusterData = readJSON(
    "cluster-data", []
);


// ==========================================
// CHECK CHART.JS
// ==========================================

if (typeof Chart === "undefined") {

    console.error("Chart.js is not loaded.");

} else {

    Chart.defaults.font.family = "Inter, sans-serif";
    Chart.defaults.color = "#8490a5";


    // ======================================
    // 1. NETWORK ACTIVITY DISTRIBUTION
    // ======================================

    const activityCanvas =
        document.getElementById("activityChart");

    if (activityCanvas) {

        new Chart(activityCanvas, {

            type: "doughnut",

            data: {

                labels: [
                    "Normal Activities",
                    "Detected Anomalies"
                ],

                datasets: [{

                    data: [
                        normalCount,
                        anomalyCount
                    ],

                    backgroundColor: [
                        "#28b77a",
                        "#ef5350"
                    ],

                    borderWidth: 4,
                    borderColor: "#ffffff",
                    hoverOffset: 8

                }]

            },

            options: {

                responsive: true,
                maintainAspectRatio: false,
                cutout: "70%",

                plugins: {

                    legend: {

                        position: "bottom",

                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }

                    }

                }

            }

        });

    }


    // ======================================
    // 2. ATTACK CATEGORY DISTRIBUTION
    // ======================================

    const attackCanvas =
        document.getElementById("attackChart");

    if (attackCanvas) {

        new Chart(attackCanvas, {

            type: "bar",

            data: {

                labels: attackLabels,

                datasets: [{

                    label: "Records",

                    data: attackCounts,

                    backgroundColor: [
                        "#3478f6",
                        "#7658e8",
                        "#24b77c",
                        "#f2a343",
                        "#e95e65",
                        "#31a9c9",
                        "#a77be8",
                        "#e58d47",
                        "#5d91d8",
                        "#7cba78"
                    ],

                    borderRadius: 6,
                    borderSkipped: false

                }]

            },

            options: {

                responsive: true,
                maintainAspectRatio: false,

                plugins: {

                    legend: {
                        display: false
                    }

                },

                scales: {

                    y: {

                        beginAtZero: true,

                        grid: {
                            color: "#edf0f5"
                        }

                    },

                    x: {

                        grid: {
                            display: false
                        }

                    }

                }

            }

        });

    }


    // ======================================
    // 3. TRAFFIC ANALYSIS
    // ======================================

    const trafficCanvas =
        document.getElementById("trafficChart");

    if (trafficCanvas) {

        new Chart(trafficCanvas, {

            type: "bar",

            data: {

                labels: [
                    "Normal Activities",
                    "Potential Anomalies"
                ],

                datasets: [{

                    label: "Number of Records",

                    data: [
                        normalCount,
                        anomalyCount
                    ],

                    backgroundColor: [
                        "#28b77a",
                        "#ef5350"
                    ],

                    borderRadius: 8,
                    barThickness: 65

                }]

            },

            options: {

                responsive: true,
                maintainAspectRatio: false,

                plugins: {

                    legend: {
                        display: false
                    }

                },

                scales: {

                    y: {

                        beginAtZero: true,

                        grid: {
                            color: "#edf0f5"
                        }

                    },

                    x: {

                        grid: {
                            display: false
                        }

                    }

                }

            }

        });

    }


    // ======================================
    // 4. DBSCAN CLUSTER VISUALIZATION
    // ======================================

    const clusterCanvas =
        document.getElementById("clusterChart");

    if (clusterCanvas) {

        const normalPoints = [];
        const anomalyPoints = [];


        if (Array.isArray(clusterData)) {

            clusterData.forEach(point => {

                const x = Number(point.PCA_X);
                const y = Number(point.PCA_Y);

                const cluster = Number(
                    point.DBSCAN_Result
                );


                if (
                    !Number.isFinite(x) ||
                    !Number.isFinite(y)
                ) {
                    return;
                }


                const item = {
                    x: x,
                    y: y
                };


                if (cluster === -1) {

                    anomalyPoints.push(item);

                } else {

                    normalPoints.push(item);

                }

            });

        }


        new Chart(clusterCanvas, {

            type: "scatter",

            data: {

                datasets: [

                    {

                        label: "Cluster Points",

                        data: normalPoints,

                        backgroundColor: "#3478f6",

                        pointRadius: 3

                    },

                    {

                        label: "Noise / Potential Anomalies",

                        data: anomalyPoints,

                        backgroundColor: "#ef5350",

                        pointRadius: 4

                    }

                ]

            },

            options: {

                responsive: true,
                maintainAspectRatio: false,

                plugins: {

                    legend: {

                        position: "bottom",

                        labels: {
                            usePointStyle: true,
                            padding: 20
                        }

                    }

                },

                scales: {

                    x: {

                        type: "linear",

                        position: "bottom",

                        title: {

                            display: true,

                            text: "PCA Component 1"

                        },

                        grid: {
                            color: "#edf0f5"
                        }

                    },

                    y: {

                        title: {

                            display: true,

                            text: "PCA Component 2"

                        },

                        grid: {
                            color: "#edf0f5"
                        }

                    }

                }

            }

        });

    }

}


// ==========================================
// SIDEBAR ACTIVE MENU
// ==========================================

const navLinks =
    document.querySelectorAll(".nav-link");

navLinks.forEach(link => {

    link.addEventListener("click", function () {

        navLinks.forEach(item => {

            item.classList.remove("active");

        });

        this.classList.add("active");

    });

});


// ==========================================
// DEBUG INFORMATION
// ==========================================

console.log("Dashboard JavaScript loaded.");

console.log("Normal records:", normalCount);

console.log("Anomaly records:", anomalyCount);

console.log("Attack categories:", attackLabels.length);

console.log("DBSCAN records:", clusterData.length);
