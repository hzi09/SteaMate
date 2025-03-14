from django.shortcuts import render, redirect
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import status
from .models import User, UserPreferredGame, Game
from .serializers import (CreateUserSerializer, UserUpdateSerializer,
                          SteamSignupSerializer)
from rest_framework.permissions import AllowAny, IsAuthenticated, IsAdminUser
from rest_framework import permissions
from django.shortcuts import get_object_or_404
from django.conf import settings
import urllib.parse
from rest_framework_simplejwt.tokens import RefreshToken
import requests
from rest_framework_simplejwt.exceptions import InvalidToken
from rest_framework_simplejwt.exceptions import TokenError
import os
from dotenv import load_dotenv
from .utils import fetch_steam_library, get_or_create_game, get_or_create_genre
from rest_framework_simplejwt.authentication import JWTAuthentication
from django.urls import reverse
from django.utils.http import urlsafe_base64_decode, urlsafe_base64_encode
from django.utils.encoding import force_bytes, force_str
from django.contrib.auth.tokens import default_token_generator
from django.core.mail import send_mail


load_dotenv()
STEAM_API_KEY = os.getenv("STEAM_API_KEY")


class SignupAPIView(APIView):
    """일반 사용자 회원가입 API"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        serializer = CreateUserSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            user = serializer.save()
            
            # 이메일 인증 토큰 생성
            uid = urlsafe_base64_encode(force_bytes(user.pk))
            token = default_token_generator.make_token(user)
            verification_url = request.build_absolute_uri(
                reverse("account:verify-email", kwargs = {'uidb64': uid, 'token': token})
            )
            
            # 이메일 전송
            send_mail(
                subject="이메일 인증",
                message=f"이메일 인증을 위해 다음 링크를 클릭해주세요: {verification_url}",
                from_email=settings.EMAIL_HOST_USER,
                recipient_list=[user.email],
                fail_silently=False
            )
            return Response({
                "message":"회원가입 완료. 이메일을 확인하고 인증을 완료하세요.",
                "email_verification_url":verification_url
                }, status=status.HTTP_201_CREATED)


class EmailVerifyAPIView(APIView):
    """이메일 인증 API"""
    permission_classes = [AllowAny]
    
    def get(self, request, uidb64, token):
        try:
            # uidb64를 다시 pk 값으로 돌려 user 확인
            uid = force_str(urlsafe_base64_decode(uidb64))
            user = get_object_or_404(User, pk=uid)
            
            if default_token_generator.check_token(user, token):
                user.is_verified = True
                user.save()
                return Response({"message":"이메일 인증이 완료되었습니다."}, status=status.HTTP_200_OK)
            else:
                return Response({"error":"유효하지 않은 토큰입니다."}, status=status.HTTP_400_BAD_REQUEST)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            return Response({"error":"잘못된 요청입니다."}, status=status.HTTP_400_BAD_REQUEST)

class SteamLoginAPIView(APIView):
    """Steam OpenID 로그인 요청"""
    permission_classes = [AllowAny]

    def get(self, request):
        if request.user.is_authenticated:
            user = request.user
            if user.steam_id:
                return Response({"error": "이미 Steam 계정 연동이 되어 있습니다."}, status=status.HTTP_400_BAD_REQUEST)
        
        """GET 요청 시 Steam 로그인 페이지로 리디렉션"""
        steam_openid_url = "https://steamcommunity.com/openid/login"
        
        params = {
            "openid.ns": "http://specs.openid.net/auth/2.0",
            "openid.mode": "checkid_setup",
            "openid.return_to": f"{settings.SITE_URL}/api/v1/account/steam-callback/",
            "openid.realm": settings.SITE_URL,
            "openid.identity": "http://specs.openid.net/auth/2.0/identifier_select",
            "openid.claimed_id": "http://specs.openid.net/auth/2.0/identifier_select",
        }

        steam_login_url = f"{steam_openid_url}?{urllib.parse.urlencode(params)}"
        return redirect(steam_login_url)
            
class SteamCallbackAPIView(APIView):
    """Steam 로그인 Callback API (Steam ID 검증)"""
    permission_classes = [AllowAny]
    authentication_classes = [JWTAuthentication]
    
    def get(self, request):
        """Steam 로그인 성공 후, OpenID 검증"""

        # GET 파라미터를 dict 형태로 변환
        openid_params = request.GET
        
        # 필수 OpenID 파라미터 유지
        steam_openid_params = {
            "openid.ns": openid_params.get("openid.ns", ""),
            "openid.mode": "check_authentication",
            "openid.op_endpoint": openid_params.get("openid.op_endpoint", ""),
            "openid.claimed_id": openid_params.get("openid.claimed_id", ""),
            "openid.identity": openid_params.get("openid.identity", ""),
            "openid.return_to": openid_params.get("openid.return_to", ""),
            "openid.response_nonce": openid_params.get("openid.response_nonce", ""),
            "openid.assoc_handle": openid_params.get("openid.assoc_handle", ""),
            "openid.signed": openid_params.get("openid.signed", ""),
            "openid.sig": openid_params.get("openid.sig", ""),
        }

        steam_openid_url = "https://steamcommunity.com/openid/login"

        # Steam OpenID 검증 요청 (POST 사용)
        response = requests.post(steam_openid_url, data=steam_openid_params)

        # Steam 응답 처리
        response_text = response.text.strip()
        print("Steam OpenID 응답 (첫 50자):", response_text[:50])

        # Steam 인증 실패 시
        if "is_valid:true" not in response_text:
            return Response(
                {"error": "Steam 인증 실패", "steam_response": response_text[:200]},
                status=status.HTTP_400_BAD_REQUEST
            )

        # Steam ID 추출 (예외 처리 강화)
        steam_id_url = openid_params.get("openid.claimed_id", "")
        if not steam_id_url or not steam_id_url.startswith("https://steamcommunity.com/openid/id/"):
            return Response({"error": "잘못된 Steam ID 응답"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            steam_id = steam_id_url.split("/")[-1]
            if not steam_id.isdigit():
                raise ValueError("Steam ID가 숫자가 아닙니다.")
        except Exception as e:
            return Response({"error": f"Steam ID 처리 중 오류 발생: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)

        # DB에서 해당 Steam ID가 존재하는지 확인
        if request.user.is_authenticated:
            user = request.user
            if user.steam_id:
                return Response({"error": "이미 Steam 계정 연동이 되어 있습니다."}, status=status.HTTP_400_BAD_REQUEST)
            
            if User.objects.filter(steam_id=steam_id).exists():
                return Response({"error": "이미 다른 계정에 연동된 Steam ID입니다."}, status=status.HTTP_400_BAD_REQUEST)

            user.steam_id = steam_id
            user.save()
            return Response({"message": "Steam 계정 연동 완료"}, status=status.HTTP_200_OK)
        
        
        user = User.objects.filter(steam_id=steam_id).first()
        print(f"result: {user}")
        
        if user:
            # 기존 회원이면 JWT 발급 후 로그인 처리
            refresh = RefreshToken.for_user(user)
            return Response({
                "message": "Steam 로그인 성공",
                "access": str(refresh.access_token),
                "refresh": str(refresh),
                "user_id": user.id,
                "redirect_url": "/"  # 홈으로 리다이렉트
            }, status=status.HTTP_200_OK)
        
        # 신규 회원이면 추가 정보 입력 필요 → 회원가입 페이지로 리다이렉트
        return Response({
            "message": "Steam 인증 성공. 추가 정보 입력이 필요합니다.",
            "steam_id": steam_id,
            "needs_update": True,
            "redirect_url": "/signup"  # 회원가입 페이지로 이동
        }, status=status.HTTP_201_CREATED)



class SteamSignupAPIView(APIView):
    """Steam 회원가입 (추가 정보 입력)"""
    permission_classes = [AllowAny]

    def post(self, request):
        """Steam 회원가입: 추가 정보 입력 후 계정 생성"""
        serializer = SteamSignupSerializer(data=request.data)
        if serializer.is_valid(raise_exception=True):
            user = serializer.save()
            
            appids, titles, playtimes = fetch_steam_library(user.steam_id)
            
            if not appids:
                print(f"Steam 라이브러리 불러오기 실패 또는 빈 데이터 (steam_id: {user.steam_id})")
            
            user_preferred_games = []
            
            for i in range(len(appids)):
                game = get_or_create_game(appid=appids[i])
                if game:
                    user_preferred_games.append(UserPreferredGame(user=user, game=game, playtime=playtimes[i]))
            
            try:
                if user_preferred_games:
                    UserPreferredGame.objects.bulk_create(user_preferred_games)
            except Exception as e:
                print(f"UserPreferredGame 생성 오류: {str(e)}")
                

            # JWT 토큰 발급
            refresh = RefreshToken.for_user(user)
            response_data = serializer.data
            return Response({
                **serializer.data,  # 기존 serializer 데이터 유지
                "message": "Steam 회원가입 완료",
                "access": str(refresh.access_token),
                "refresh": str(refresh),
                "user_id": user.id,
                "redirect_url": "/"
            }, status=status.HTTP_201_CREATED)





class MyPageAPIView(APIView):
    """사용자 정보 조회, 수정, 삭제 API"""
    def get_permissions(self):
        """요청 방식(GET, PUT, DELETE)에 따라 다른 권한을 적용"""
        if self.request.method == "GET":
            return [permissions.AllowAny()]
        return [permissions.IsAuthenticated()]
    
    def get_user(self, pk):
        return get_object_or_404(User, pk=pk)
        
    def get(self, request, pk):
        """사용자 정보 조회"""
        user = self.get_user(pk)
        serializer = UserUpdateSerializer(user)
        data = serializer.data
        
        # Steam API로 사용자 정보 가져오기
        if user.steam_id:
            steam_url = f"http://api.steampowered.com/ISteamUser/GetPlayerSummaries/v2/?key={STEAM_API_KEY}&steamids={user.steam_id}"
            
            try:
                response = requests.get(steam_url)
                response.raise_for_status()
                response_data = response.json()

                if "response" in response_data and "players" in response_data["response"]:
                    steam_data = response_data["response"]["players"][0]

                    data["steam_profile"] = {
                        "personaname": steam_data.get("personaname"),
                        "profileurl": steam_data.get("profileurl"),
                        "avatar": steam_data.get("avatar"),
                        "country": steam_data.get("loccountrycode"),
                    }
                    # 선호 게임이 없다면 라이브러리 전체를 가져와 저장
                    if not UserPreferredGame.objects.filter(user=user).exists():
                        appids, titles, playtimes = fetch_steam_library(user.steam_id)
                        
                        user_preferred_games = []
                        
                        for i in range(len(appids)):
                            game = get_or_create_game(appid=appids[i])
                            if game:
                                user_preferred_games.append(UserPreferredGame(user=user, game=game, playtime=playtimes[i]))
                                
                        try:
                            if user_preferred_games:
                                UserPreferredGame.objects.bulk_create(user_preferred_games)
                        except Exception as e:
                            print(f"UserPreferredGame 생성 오류: {str(e)}")

                else:
                    data["steam_profile_error"] = "Steam 프로필 정보를 가져오지 못했습니다."
            
            except Exception as e:
                data["steam_profile_error"] = f"Steam API 호출 오류: {str(e)}"
        data["preferred_genre"] = [genre.genre_name for genre in user.preferred_genre.all()]
        data["preferred_game"] = [game.title for game in user.preferred_game.all()]
        return Response(data, status=status.HTTP_200_OK)


    def put(self, request,pk):
        """사용자 정보 수정"""
        if pk != request.user.pk:
            return Response({"error": "You do not have permission to this page"},status=status.HTTP_403_FORBIDDEN)
        user = self.get_user(request.user.pk)
        serializer = UserUpdateSerializer(user, data=request.data)
        if serializer.is_valid(raise_exception=True):
            serializer.save()
            return Response(serializer.data, status = status.HTTP_200_OK)
    
    def delete(self, request, pk):
        """사용자 탈퇴 및 정보 삭제"""
        if pk != request.user.pk:
            return Response({"error": "You do not have permission to delete this user"},status=status.HTTP_403_FORBIDDEN)
        
        user = self.get_user(request.user.pk)
        user.delete()
        return Response({"message":"withdrawal"},status=status.HTTP_204_NO_CONTENT)
    

class LogoutAPIView(APIView):
    """
    로그아웃 API
    """
    permission_classes = [IsAuthenticated]

    def post(self, request):
        try:
            refresh_token = request.data.get("refresh")

            # refresh_token이 없을 때
            if not refresh_token:
                return Response({"error": "Refresh token is required."}, status=status.HTTP_400_BAD_REQUEST)

            token = RefreshToken(refresh_token)
            token.blacklist()

            return Response({"detail": "Successfully logged out."}, status=status.HTTP_200_OK)

        except TokenError:
            return Response({"error": "Invalid or expired token."}, status=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            # 기타 예외 발생 시
            return Response({"error": f"Token processing error: {str(e)}"}, status=status.HTTP_400_BAD_REQUEST)