import asyncio
import json
import time
from typing import Dict, List
from agents.orchestrator_agent import OrchestratorAgent
from config.settings import settings

class VideoSearchSystem:
    def __init__(self):
        self.orchestrator = OrchestratorAgent()
        self.search_history = []
    
    async def search(self, query: str) -> Dict:
        """Main search interface"""
        print(f"\n🔍 Searching: '{query}'")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Process query through orchestrator
            result = await self.orchestrator.process_with_cache(query)
            
            processing_time = time.time() - start_time
            
            # Format response
            response = {
                'query': query,
                'success': result.success,
                'confidence': result.confidence,
                'processing_time': processing_time,
                'explanation': result.explanation,
                'results': [r.to_dict() for r in result.results],
                'metadata': result.metadata,
                'agents_used': result.metadata.get('agents_used', []),
                'total_results': len(result.results)
            }
            
            # Log summary
            self._log_search_summary(response)
            
            # Store in history
            self.search_history.append(response)
            
            return response
            
        except Exception as e:
            error_response = {
                'query': query,
                'success': False,
                'confidence': 0.0,
                'processing_time': time.time() - start_time,
                'explanation': f"Search failed: {str(e)}",
                'results': [],
                'metadata': {},
                'error': str(e)
            }
            
            print(f"❌ Search failed: {e}")
            return error_response
    
    def _log_search_summary(self, response: Dict):
        """Log search summary"""
        print(f"✅ Success: {response['success']}")
        print(f"🎯 Confidence: {response['confidence']:.2f}")
        print(f"⏱️  Time: {response['processing_time']:.2f}s")
        print(f"🤖 Agents: {', '.join(response['agents_used'])}")
        print(f"📊 Results: {response['total_results']}")
        print(f"💭 Explanation: {response['explanation']}")
        
        if response['results']:
            print("\n📋 Top Results:")
            for i, result in enumerate(response['results'][:5]):
                score = result['score']
                video_id = result['video_id']
                keyframe_id = result.get('keyframe_id', 'N/A')
                print(f"  {i+1}. {video_id}/{keyframe_id} (score: {score:.3f})")
    
    async def batch_search(self, queries: List[str]) -> List[Dict]:
        """Search multiple queries"""
        results = []
        
        print(f"\n🔄 Batch searching {len(queries)} queries...")
        
        for i, query in enumerate(queries):
            print(f"\n[{i+1}/{len(queries)}]", end=" ")
            result = await self.search(query)
            results.append(result)
            
            # Brief pause to avoid rate limiting
            await asyncio.sleep(0.1)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        if not self.search_history:
            return {"message": "No searches performed yet"}
        
        total_searches = len(self.search_history)
        successful_searches = sum(1 for s in self.search_history if s['success'])
        avg_confidence = sum(s['confidence'] for s in self.search_history) / total_searches
        avg_processing_time = sum(s['processing_time'] for s in self.search_history) / total_searches
        
        # Agent usage stats
        agent_usage = {}
        for search in self.search_history:
            for agent in search.get('agents_used', []):
                agent_usage[agent] = agent_usage.get(agent, 0) + 1
        
        return {
            'total_searches': total_searches,
            'success_rate': successful_searches / total_searches,
            'average_confidence': avg_confidence,
            'average_processing_time': avg_processing_time,
            'agent_usage': agent_usage,
            'cache_hits': sum(1 for s in self.search_history if 'cache_hit' in s.get('metadata', {}))
        }

async def main():
    """Main function for testing the system"""
    # Initialize system
    search_system = VideoSearchSystem()
    
    print("🚀 Video Search AI Agent System Started")
    print("=" * 60)
    
    # Test queries
    test_queries = [
        "tìm keyframe có một người đàn ông mặc áo đỏ đang cầm điện thoại quay phim những người chạy xe đạp"
    ]
    
    # Interactive mode
    while True:
        print("\n" + "="*60)
        print("Options:")
        print("1. Enter custom query")
        print("2. Test with sample queries")
        print("3. View system stats")
        print("4. Exit")
        
        choice = input("\nChọn option (1-4): ").strip()
        
        if choice == '1':
            query = input("Nhập query: ").strip()
            if query:
                await search_system.search(query)
        
        elif choice == '2':
            print("\n🧪 Testing with sample queries...")
            await search_system.batch_search(test_queries)
        
        elif choice == '3':
            stats = search_system.get_stats()
            print("\n📊 System Statistics:")
            print(json.dumps(stats, indent=2, ensure_ascii=False))
        
        elif choice == '4':
            print("👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid option")

if __name__ == "__main__":
    # Check required settings
    required_settings = ['GOOGLE_API_KEY', 'QDRANT_HOST', 'QDRANT_PORT', 
                        'METADATA_KEYFRAME_OBJECT_DB_PATH', 'QDRANT_VIDEO_COLLECTION_NAME', 
                        'QDRANT_KEYWORD_COLLECTION_NAME']
    
    missing_settings = []
    for setting in required_settings:
        if not hasattr(settings, setting):
            missing_settings.append(setting)
    
    if missing_settings:
        print(f"❌ Missing settings: {', '.join(missing_settings)}")
        print("Please configure these in config/settings.py")
        exit(1)
        
    errors = settings.validate_settings()
    if errors:
        print(errors)
        exit(1)
        
    # Run main interface
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 System stopped by user")
    except Exception as e:
        print(f"❌ System error: {e}")