import cv2
import mediapipe as mp

# Inicializa o MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7)

mp_draw = mp.solutions.drawing_utils

# Abrir webcam
cap = cv2.VideoCapture(0)

def is_fist(landmarks):
    # Verifica se a mão está fechada (punho cerrado).

    tips_ids = [8, 12, 16, 20]   # Pontas dos dedos
    pip_ids = [6, 10, 14, 18]   # Juntas médias

    closed_fingers = 0

    for tip, pip in zip(tips_ids, pip_ids):
        if landmarks[tip].y > landmarks[pip].y:
            closed_fingers += 1

    return closed_fingers >= 4  # Punho cerrado

def is_dark_environment(frame, threshold=80):
    # Detecta se o ambiente está escuro baseado na média de luminosidade.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = gray.mean()
    return mean_brightness < threshold, mean_brightness

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Erro ao capturar imagem.")
        break

    # Redimensiona o vídeo
    frame = cv2.resize(frame, (500, 500))

    # Verifica luminosidade
    dark, brightness = is_dark_environment(frame)

    # Converte BGR para RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if is_fist(hand_landmarks.landmark) and dark:
                # Exibe alerta apenas em ambiente escuro
                cv2.putText(frame, "SOCORRO DETECTADO!", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # Exibe no canto inferior esquerdo o status de luminosidade
    status = "Baixa" if dark else "Alta"
    cv2.putText(frame, f"Luminosidade: {status}", (10, 490),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Punho Cerrado Pedido de Socorro", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
