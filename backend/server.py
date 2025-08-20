from flask import Flask, request, jsonify
from flask_cors import CORS
from main import parse_csv, Boat, build_boat
import csv
import os
from shutil import copyfile
import gspread
import pandas as pd

app = Flask(__name__)
CORS(app)

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROSTER_FILE = os.path.join(BASE_DIR, 'roster.csv')
ACTIVE_ROSTER_FILE = os.path.join(BASE_DIR, 'roster-active.csv')
SERVICE_ACCOUNT_FILE = os.path.join(BASE_DIR, 'google-sheets-credentials.json')

# Google Sheet URL and sheet name for attendance
SPREADSHEET_URL = 'https://docs.google.com/spreadsheets/d/1dqwB_jHAwzj5NmJdbe9SytnnqEpaDZfbGjkkwcIDtE0/edit'
SHEET_NAME = 'Test'

# Google Sheet ID and GID for lineup
LINEUP_SHEET_ID = '1SZMIH8hu3vMTnZfVpRkCJLnR-Jsbr8JMiNn0L1MRWoI'
LINEUP_SHEET_GID = '0'


# Initialize roster-active.csv if it doesn't exist
def initialize_active_roster():
    if not os.path.exists(ACTIVE_ROSTER_FILE):
        try:
            copyfile(ROSTER_FILE, ACTIVE_ROSTER_FILE)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: {ROSTER_FILE} not found. Please ensure it exists.")


def get_google_sheet_data():
    try:
        # Authenticate with Google Sheets
        gc = gspread.service_account(filename=SERVICE_ACCOUNT_FILE)
        # Open the spreadsheet by URL
        sh = gc.open_by_url(SPREADSHEET_URL)
        # Select the worksheet/tab
        worksheet = sh.worksheet(SHEET_NAME)
        # Get all values from the sheet
        return worksheet.get_all_values()
    except Exception as exc:
        raise Exception(f"Failed to scrape Google Sheet: {str(exc)}")


def parse_attendance(data):
    present = []
    absent = []

    for row in data:
        for i, cell in enumerate(row):
            cell = cell.strip()
            if not cell:
                continue

            # Case 1: "TRUE"/"FALSE" then a name in the next cell
            if cell.upper() in ("TRUE", "FALSE"):
                if i + 1 < len(row) and row[i + 1].strip():
                    name = row[i + 1].strip()
                    if cell.upper() == "FALSE":
                        present.append(name)
                    else:
                        absent.append(name)

            # Case 2: Name followed by "TRUE"/"FALSE"
            elif i + 1 < len(row) and row[i + 1].strip().upper() in ("TRUE", "FALSE"):
                name = cell
                status = row[i + 1].strip().upper()
                if status == "FALSE":
                    present.append(name)
                else:
                    absent.append(name)

    return present, absent


def import_lineup():
    try:
        # Construct the export URL for the lineup Google Sheet
        url = f"https://docs.google.com/spreadsheets/d/{LINEUP_SHEET_ID}/export?format=csv&gid={LINEUP_SHEET_GID}"
        # Load the sheet into a DataFrame
        df = pd.read_csv(url)
        # Validate required columns
        required_columns = ['Name', 'Gender', 'Side', 'Weight']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Google Sheet must contain columns: {', '.join(required_columns)}")
        # Save to roster.csv and roster-active.csv
        df.to_csv(ROSTER_FILE, index=False)
        df.to_csv(ACTIVE_ROSTER_FILE, index=False)
        # Parse the CSV to return serialized data
        people = parse_csv(ROSTER_FILE)
        serialized = [
            {
                'name': p.name,
                'gender': p.gender,
                'weight': p.weight,
                'side': p.side,
                'active': True  # All imported people are active
            }
            for p in people
        ]
        return serialized
    except Exception as exc:
        raise Exception(f"Failed to import lineup: {str(exc)}")


@app.route('/people', methods=['GET'])
def list_people():
    try:
        initialize_active_roster()
        all_people = parse_csv(ROSTER_FILE)
        active_people = parse_csv(ACTIVE_ROSTER_FILE)
        active_names = {p.name for p in active_people}

        serialized = [
            {
                'name': p.name,
                'gender': p.gender,
                'weight': p.weight,
                'side': p.side,
                'active': p.name in active_names
            }
            for p in all_people
        ]
        return jsonify({'people': serialized, 'count': len(serialized)}), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/people/<name>/toggle', methods=['POST'])
def toggle_person(name):
    try:
        initialize_active_roster()
        all_people = parse_csv(ROSTER_FILE)
        active_people = parse_csv(ACTIVE_ROSTER_FILE)
        active_names = {p.name for p in active_people}

        if name not in {p.name for p in all_people}:
            return jsonify({'error': f"Person '{name}' not found in roster"}), 404

        if name in active_names:
            # Remove person from active roster
            active_people = [p for p in active_people if p.name != name]
        else:
            # Add person to active roster
            person = next(p for p in all_people if p.name == name)
            active_people.append(person)

        # Write updated active roster to file
        with open(ACTIVE_ROSTER_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Name', 'Gender', 'Side', 'Weight'])
            writer.writeheader()
            for p in active_people:
                writer.writerow({
                    'Name': p.name,
                    'Gender': p.gender,
                    'Side': p.side,
                    'Weight': p.weight
                })

        return jsonify({'message': f"Person '{name}' toggled successfully", 'active': name not in active_names}), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/people/toggle-all', methods=['POST'])
def toggle_all_people():
    try:
        initialize_active_roster()
        payload = request.get_json(force=True) or {}
        set_active = payload.get('active', True)

        if set_active:
            # Copy all people from roster.csv to roster-active.csv
            copyfile(ROSTER_FILE, ACTIVE_ROSTER_FILE)
            message = "All people activated"
        else:
            # Clear roster-active.csv (write only header)
            with open(ACTIVE_ROSTER_FILE, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=['Name', 'Gender', 'Side', 'Weight'])
                writer.writeheader()
            message = "All people deactivated"

        return jsonify({'message': message}), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/people/scrape-attendance', methods=['POST'])
def scrape_attendance():
    try:
        initialize_active_roster()
        all_people = parse_csv(ROSTER_FILE)
        all_names = {p.name for p in all_people}

        # Scrape Google Sheet data
        data = get_google_sheet_data()
        present, absent = parse_attendance(data)

        # Validate names against roster.csv
        invalid_names = [name for name in present + absent if name not in all_names]
        if invalid_names:
            return jsonify({'error': f"Invalid names in Google Sheet: {', '.join(invalid_names)}"}), 400

        # Update roster-active.csv with present people
        active_people = [p for p in all_people if p.name in present]

        # Write updated active roster to file
        with open(ACTIVE_ROSTER_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Name', 'Gender', 'Side', 'Weight'])
            writer.writeheader()
            for p in active_people:
                writer.writerow({
                    'Name': p.name,
                    'Gender': p.gender,
                    'Side': p.side,
                    'Weight': p.weight
                })

        return jsonify({
            'message': 'Attendance scraped and roster updated successfully',
            'present': present,
            'absent': absent
        }), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/people/import-lineup', methods=['POST'])
def import_lineup_endpoint():
    try:
        # Import lineup and get serialized data
        people = import_lineup()
        return jsonify({
            'message': 'Lineup imported successfully',
            'people': people
        }), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/assignments', methods=['POST'])
def create_assignments():
    try:
        initialize_active_roster()
        payload = request.get_json(force=True) or {}
        boats_payload = payload.get('boats', [])

        boats = []
        for boat in boats_payload:
            size = int(boat.get('size', 0))
            gender = boat.get('gender', 'Mixed')
            boats.append(Boat(size=size, gender=gender))

        people = parse_csv(ACTIVE_ROSTER_FILE)  # Use active roster for assignments

        result = build_boat(people, boats)

        # Serialize dataclasses to dicts for JSON response
        serialized = {
            'first_half': {},
            'second_half': {}
        }

        # Serialize first_half
        for idx in result['first_half']:
            left_side, right_side = result['first_half'][idx]
            serialized['first_half'][str(idx)] = {
                'left': [
                    {
                        'name': p.name,
                        'gender': p.gender,
                        'weight': p.weight,
                        'side': p.side,
                    }
                    for p in left_side
                ],
                'right': [
                    {
                        'name': p.name,
                        'gender': p.gender,
                        'weight': p.weight,
                        'side': p.side,
                    }
                    for p in right_side
                ],
            }

        # Serialize second_half
        for idx in result['second_half']:
            left_side, right_side = result['second_half'][idx]
            serialized['second_half'][str(idx)] = {
                'left': [
                    {
                        'name': p.name,
                        'gender': p.gender,
                        'weight': p.weight,
                        'side': p.side,
                    }
                    for p in left_side
                ],
                'right': [
                    {
                        'name': p.name,
                        'gender': p.gender,
                        'weight': p.weight,
                        'side': p.side,
                    }
                    for p in right_side
                ],
            }

        return jsonify({'assignments': serialized}), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)