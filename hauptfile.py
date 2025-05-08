import csv
import os
import yfinance as yf
from tabulate import tabulate
import matplotlib.pyplot as plt

# ==== Einstellungen ====
CSV_FILE = "portfolio.csv"

# ==== Portfolio aus CSV laden ====
def load_portfolio(file_path):
    portfolio = []
    
    if not os.path.exists(file_path):
        print(f"Datei {file_path} nicht gefunden!")
        return portfolio

    with open(file_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ticker = row["Ticker"].strip()
                quantity = float(row["Menge"])
                buy_price = float(row["Kaufpreis"])
                portfolio.append({
                    "ticker": ticker,
                    "quantity": quantity,
                    "buy_price": buy_price
                })
            except KeyError:
                print("CSV Fehler: Bitte stelle sicher, dass die Spalten 'Ticker', 'Menge' und 'Kaufpreis' heißen.")
                break
            except ValueError:
                print(f"Ungültiger Wert in Zeile: {row}")
                continue

    return portfolio

# ==== Aktuellen Preis abrufen ====
def get_current_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        price = stock.history(period="1d")["Close"].iloc[-1]
        return price
    except:
        return None

# ==== Portfolio analysieren und anzeigen ====
def analyze_portfolio(portfolio):
    table = []
    total_value = 0
    total_gain = 0
    pie_labels = []
    pie_values = []

    for asset in portfolio:
        ticker = asset["ticker"]
        quantity = asset["quantity"]
        buy_price = asset["buy_price"]
        current_price = get_current_price(ticker)

        if current_price is None:
            table.append([ticker, quantity, buy_price, "Fehler", "-", "-"])
            continue

        total = current_price * quantity
        gain = (current_price - buy_price) * quantity

        total_value += total
        total_gain += gain

        table.append([
            ticker,
            quantity,
            round(buy_price, 2),
            round(current_price, 2),
            round(total, 2),
            round(gain, 2)
        ])

        pie_labels.append(ticker)
        pie_values.append(total)

    headers = ["Ticker", "Menge", "Kaufpreis", "Aktuell", "Wert", "Gewinn/Verlust"]
    print("\n📊 --- Portfolio Übersicht ---")
    print(tabulate(table, headers=headers, tablefmt="pretty"))
    print(f"\nGesamtwert des Portfolios: € {round(total_value, 2)}")
    print(f"Gesamter Gewinn/Verlust: € {round(total_gain, 2)}")

    # Pie Chart anzeigen
    if pie_values:
        show_pie_chart(pie_labels, pie_values)

# ==== Pie Chart zeichnen ====
def show_pie_chart(labels, values):
    plt.figure(figsize=(6,6))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Portfolio Verteilung")
    plt.axis("equal")
    plt.show()

# ==== Hauptprogramm ====
def main():
    print("📥 Lade Portfolio aus CSV...")
    portfolio = load_portfolio(CSV_FILE)

    if not portfolio:
        print("Keine Daten gefunden oder Fehler beim Laden.")
        return

    analyze_portfolio(portfolio)

if __name__ == "__main__":
    main()
