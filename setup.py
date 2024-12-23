from setuptools import setup, find_packages

setup(
    name="mlship",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "mlship.ui": [
            "templates/*.html",
            "static/js/*.js",
        ],
    },
    zip_safe=False,
) 