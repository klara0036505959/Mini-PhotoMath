from photomath_app_flask.predict_on_image import *
import argparse


def predict(args):
    pred, sol = do_prediction(args.img_path, args.model_path)
    
    print("Prediction: " + str(pred))
    print("Evaluated expression: " + str(sol))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Params")
    parser.add_argument(
        "--model_path",
        type=str,
        default="trained_models/my_model_good_final.h5",
        help="Path to the saved model",
    )

    parser.add_argument(
        "--img_path",
        type=str,
        required=True,
        help="Path to the image containing a math expression",
    )
    args = parser.parse_args()
    predict(args)