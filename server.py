from flask import Flask, request, jsonify
from flask_cors import CORS
from main import parse_csv, Boat, build_boat


app = Flask(__name__)
CORS(app)


@app.route('/people', methods=['GET'])
def list_people():
    try:
        people = parse_csv('roster1.csv')
        serialized = [
            {
                'name': p.name,
                'gender': p.gender,
                'weight': p.weight,
                'side': p.side,
            }
            for p in people
        ]
        return jsonify({'people': serialized, 'count': len(serialized)}), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/assignments', methods=['POST'])
def create_assignments():
    try:
        payload = request.get_json(force=True) or {}
        boats_payload = payload.get('boats', [])

        boats = []
        for boat in boats_payload:
            size = int(boat.get('size', 0))
            gender = boat.get('gender', 'Mixed')
            boats.append(Boat(size=size, gender=gender))

        people = parse_csv('roster1.csv')

        result = build_boat(people, boats)

        # Serialize dataclasses to dicts for JSON response
        serialized = {}
        for idx, (left_side, right_side) in result.items():
            serialized[str(idx)] = {
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


