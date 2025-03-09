from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated

from .models import ChatSession, ChatMessage
from .serializers import ChatSessionSerializer, ChatMessageSerializer

from .utils import chatbot_call, bring_session_history


# Create your views here.
class ChatSessionAPIView(APIView):

    # 인증되지 않은 유저가 접근하면 401에러를 반환
    permission_classes = [IsAuthenticated]

    # 세션 목록 조회
    def get(self, request):
        sessions = ChatSession.objects.all()
        serializer = ChatSessionSerializer(sessions, many=True)
        
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    # 세션 생성
    def post(self, request):
        serializer = ChatSessionSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save(user_id = request.user)
            return Response(serializer.data, status=status.HTTP_200_OK)


class ChatMessageAPIView(APIView):

    # 인증되지 않은 유저가 접근하면 401에러를 반환
    permission_classes = [IsAuthenticated]

    # 세션 내역 조회
    def get(self, request, session_id):
        session = ChatSession.objects.get(pk=session_id)
        # RDB에 있는 대화 내역을 메모리에 저장하는 함수
        # 지금은 대화 내역을 불러오고 30분이 지나면 메모리에서 삭제 됨
        # 추후 대화 내역을 저장하고 30분이 지나도 메모리에 남아있도록 수정 필요(튜터님께 여쭤보기)
        bring_session_history(session_id)
        messages = session.chat_messages.all()
        serializer = ChatMessageSerializer(messages, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    # 대화 내역 생성
    def post(self, request, session_id):
        session = ChatSession.objects.get(pk=session_id)
        serializer = ChatMessageSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            # 선호장르 가져오기
            genre = [genre.genre_name for genre in request.user.preferred_genre.all()]
            # 선호 게임 정보 가져오기(appid, game_title)
            appid = [game.appid for game in request.user.preferred_game.all()]
            game = [game.title for game in request.user.preferred_game.all()]
            # 챗봇 메시지 생성
            chatbot_message = chatbot_call(request.data["user_message"], session_id, genre=genre, game=game, appid=appid)
            serializer.save(session_id=session, chatbot_message=chatbot_message)
            return Response(serializer.data, status=status.HTTP_200_OK)