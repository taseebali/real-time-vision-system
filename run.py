"""
Run the optimized Blind Assistant with performance monitoring
"""

from src.core.optimized_assistant import OptimizedAssistant

def main():
    print("\n" + "="*60)
    print("OPTIMIZED BLIND ASSISTANT - PERFORMANCE TEST")
    print("="*60)
    
    # Get camera IP
    camera_ip = input("\nEnter phone IP address (or press Enter for 192.168.1.100): ").strip()
    if not camera_ip:
        camera_ip = "192.168.1.100"
    
    # Ask about captioning
    enable_caption = input("Enable scene captioning? (y/n, default: y): ").strip().lower()
    enable_captioning = enable_caption != 'n'
    
    # Ask about display
    show_display = input("Show visual display? (y/n, default: y): ").strip().lower()
    show_display_bool = show_display != 'n'
    
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"Camera IP: {camera_ip}")
    print(f"Scene Captioning: {'ENABLED' if enable_captioning else 'DISABLED'}")
    print(f"Visual Display: {'ENABLED' if show_display_bool else 'DISABLED'}")
    print("="*60)
    print("\nPress Ctrl+C to stop")
    print("Press 'Q' in display window to quit\n")
    
    # Initialize and start
    assistant = OptimizedAssistant(
        show_display=show_display_bool,
        camera_ip=camera_ip,
        enable_captioning=enable_captioning
    )
    
    try:
        assistant.start()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        assistant.cleanup()

if __name__ == "__main__":
    main()
