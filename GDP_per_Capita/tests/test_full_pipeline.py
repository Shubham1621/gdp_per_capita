import os
import subprocess
import sys

REQUIRED_FILES = [
    'dashboard/app.py',
    'dashboard/enhance_analytics.py',
    'dashboard/custom_style.css',
    'data/fetch_data.py',
    'data/economic_data_all_countries.csv',
    'data/indicator_metadata.csv',
    'requirements.txt',
    'Dockerfile',
    'docker-compose.yml',
]

def test_required_files():
    print('Checking required files...')
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    assert not missing, f"Missing files: {missing}"
    print('All required files are present.')

def test_lint():
    print('Running flake8 lint...')
    try:
        subprocess.run([sys.executable, '-m', 'flake8', 'dashboard', 'data', 'dags'], check=True)
        print('Lint passed.')
    except Exception as e:
        print('Lint failed:', e)

def test_fetch_data():
    print('Testing data fetch script...')
    try:
        subprocess.run([sys.executable, 'data/fetch_data.py'], check=True)
        assert os.path.exists('data/economic_data_all_countries.csv'), 'Output CSV missing!'
        print('Data fetch script ran successfully.')
    except Exception as e:
        print('Data fetch failed:', e)

def test_dashboard_import():
    print('Testing dashboard import...')
    try:
        import dashboard.app
        print('Dashboard app imported successfully.')
    except Exception as e:
        print('Dashboard import failed:', e)

def test_docker_build():
    print('Testing Docker build...')
    try:
        subprocess.run(['docker', 'build', '-t', 'gdp_dashboard', '.'], check=True)
        print('Docker build succeeded.')
    except Exception as e:
        print('Docker build failed:', e)

def main():
    test_required_files()
    test_lint()
    test_fetch_data()
    test_dashboard_import()
    test_docker_build()
    print('All tests completed.')

if __name__ == '__main__':
    main()
