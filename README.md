
---

## 📊 Universal Web App & Bot Performance Tester

A general-purpose automated testing tool to evaluate the **performance and reliability** of web apps, APIs, and bots.

It performs:

* ✅ Load Testing
* ✅ Failure Injection
* ✅ Rate-Limit Testing
* ✅ Long-Run Stability Testing

And logs:

* Response times
* CPU and RAM usage
* Success/failure stats

With automatic output to:

* 📄 `CSV`
* 📊 `PDF graphs`
* 📚 `SQLite database`

---

### 🚀 Features

* 🔁 Flexible input for any URL (API, web app, bot)
* 💣 Simulates real-world load and failure conditions
* 📈 Tracks performance over time
* 🧠 Analyzes results per test type
* 📤 Outputs PDF reports and CSV summaries

---

### 📦 Requirements

Install dependencies from the `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

### ▶️ How to Run

```bash
python universal_tester.py
```

You will be prompted to:

1. Enter one or more target URLs (comma-separated)
2. Select which tests to run (Load, Rate-Limit, etc.)

---

### 🗂 Output Files

| File                       | Description                             |
| -------------------------- | --------------------------------------- |
| `test_results.csv`         | Full log of all test runs               |
| `summary_by_test_type.csv` | Summarized results grouped by test type |
| `test_report.pdf`          | Graphs of response time, CPU, RAM       |
| `test_results.db`          | SQLite database of test logs (optional) |

---

### 📊 Example Use Cases

* Validate server scalability before deployment
* Monitor performance regressions
* Stress test rate-limiting and failure recovery
* Benchmark server resource impact

---

### 🧰 Built With

* [`requests`](https://pypi.org/project/requests/) — HTTP requests
* [`psutil`](https://pypi.org/project/psutil/) — CPU & RAM usage
* [`matplotlib`](https://pypi.org/project/matplotlib/) — Graph generation
* [`fpdf`](https://pyfpdf.github.io/) — PDF report generation
* [`pandas`](https://pypi.org/project/pandas/) — Data analysis

---

### 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

---

### 📄 License

This project is licensed under the MIT License. See `LICENSE` for details.

---
