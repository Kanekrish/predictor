# Breast Cancer Stage Predictor - Vercel

This project is structured for deployment on Vercel.

## Important

The deployed API expects the existing trained model and preprocessing artifacts:
- xgboost_model.pkl
- target_encoder.pkl
- feature_encoders.pkl

Set these as Vercel Environment Variables:
- MODEL_URL
- TARGET_ENCODER_URL
- FEATURE_ENCODERS_URL

Use RAW GitHub URLs, for example:
https://raw.githubusercontent.com/USERNAME/REPOSITORY/main/model/xgboost_model.pkl

Do not use the normal GitHub `blob` page URL.

## Existing model

If your existing model was produced by the supplied XGBOOST.py, the original code did not save `feature_encoders`. Therefore, the model cannot reliably process arbitrary web inputs unless the encoders used during training are also available.

Run the amended XGBOOST.py once to create all required artifacts if necessary. This does not change the XGBoost architecture; it saves the preprocessing mappings needed by the deployed app.

## Deployment

1. Put this project in GitHub.
2. Import the repository into Vercel.
3. Add the three environment variables.
4. Deploy.
5. Open the Vercel URL.
