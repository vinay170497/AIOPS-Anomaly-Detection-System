import random
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

# ---- CONFIG ----
NUM_LOGS = 1000000
START_DATE = datetime(2018, 1, 1)
END_DATE = datetime(2025, 1, 1)
filepath = r"D:\Agentic_Projects\Anomaly_Detection\data\log_files.csv"

methods = ['GET', 'POST', 'PUT', 'DELETE']
endpoints = [
    '/', '/home', '/login', '/logout',
    '/api/user', '/api/products', '/api/payment',
    '/admin', '/admin/config', '/search?q=test',
    '/download/file.zip'
]

status_codes = [200, 201, 204, 301, 302, 400, 401, 403, 404, 429, 500, 502, 503]

# ---- HELPERS ----

def random_date():
    delta = END_DATE - START_DATE
    random_sec = random.randint(0, int(delta.total_seconds()))
    dt = START_DATE + timedelta(seconds=random_sec)
    return dt, dt.strftime('%d/%b/%Y:%H:%M:%S +0530')


def weighted_status(hour):
    # Night anomalies (more errors)
    if hour < 6 or hour > 23:
        return random.choices(
            status_codes,
            weights=[20,2,2,2,2,10,10,10,10,5,10,5,5],
            k=1
        )[0]
    else:
        return random.choices(
            status_codes,
            weights=[60,5,5,5,5,5,5,5,5,2,3,2,1],
            k=1
        )[0]


def response_size(status):
    if status >= 400:
        return random.randint(100, 1500)
    return random.randint(500, 50000)


def latency():
    if random.random() < 0.02:
        return random.randint(2000, 10000)  # spike
    return random.randint(20, 500)


def generate_attack_endpoint():
    attacks = [
        "/login?user=admin' OR '1'='1",
        "/search?q=<script>alert(1)</script>",
        "/api/user?id=1;DROP TABLE users",
        "/admin?debug=true"
    ]
    return random.choice(attacks)


# ---- SESSION GENERATOR ----

def generate_session(ip, user):
    session_logs = []
    session_length = random.randint(3, 10)

    base_time, _ = random_date()

    for i in range(session_length):
        dt = base_time + timedelta(seconds=i * random.randint(1, 10))
        hour = dt.hour

        status = weighted_status(hour)

        log = '%s - %s [%s] "%s %s HTTP/1.1" %s %s "%s" "%s" %s\n' % (
            ip,
            user,
            dt.strftime('%d/%b/%Y:%H:%M:%S +0530'),
            random.choice(methods),
            random.choice(endpoints),
            status,
            response_size(status),
            fake.uri() if random.random() > 0.3 else "-",
            fake.user_agent(),
            latency()
        )

        session_logs.append(log)

    return session_logs


# ---- MAIN GENERATOR ----

with open(filepath, "w") as f:

    i = 0
    while i < NUM_LOGS:

        # ---- TRAFFIC SPIKE ----
        if random.random() < 0.01:
            spike_ip = fake.ipv4()
            for _ in range(random.randint(50, 200)):
                dt, ts = random_date()
                log = '%s - - [%s] "%s %s HTTP/1.1" %s %s "%s" "%s" %s\n' % (
                    spike_ip,
                    ts,
                    random.choice(methods),
                    random.choice(endpoints),
                    random.choice([200, 429, 500]),
                    random.randint(100, 5000),
                    "-",
                    fake.user_agent(),
                    random.randint(100, 2000)
                )
                f.write(log)
                i += 1
            continue

        # ---- ATTACK SESSION ----
        if random.random() < 0.02:
            ip = fake.ipv4()
            user = "admin"

            for _ in range(random.randint(5, 20)):
                dt, ts = random_date()

                log = '%s - %s [%s] "%s %s HTTP/1.1" %s %s "%s" "%s" %s\n' % (
                    ip,
                    user,
                    ts,
                    random.choice(["POST", "GET"]),
                    generate_attack_endpoint(),
                    random.choice([401, 403, 500]),
                    random.randint(100, 2000),
                    "-",
                    fake.user_agent(),
                    random.randint(500, 5000)
                )

                f.write(log)
                i += 1
            continue

        # ---- NORMAL SESSION ----
        ip = fake.ipv4()
        user = fake.user_name() if random.random() > 0.2 else "-"

        session = generate_session(ip, user)

        for log in session:
            f.write(log)
            i += 1

            if i >= NUM_LOGS:
                break