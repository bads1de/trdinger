import * as React from "react"

function cn(...classes: (string | undefined | null | false)[]): string {
  return classes.filter(Boolean).join(' ')
}

interface DropdownMenuProps {
  children: React.ReactNode
}

const DropdownMenu: React.FC<DropdownMenuProps> = ({ children }) => {
  const [isOpen, setIsOpen] = React.useState(false)
  
  return (
    <div className="relative inline-block text-left">
      {React.Children.map(children, child => 
        React.isValidElement(child) 
          ? React.cloneElement(child, { isOpen, setIsOpen } as any)
          : child
      )}
    </div>
  )
}

interface DropdownMenuTriggerProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'onClick'> {
  asChild?: boolean
  isOpen?: boolean
  setIsOpen?: (open: boolean) => void
}

const DropdownMenuTrigger = React.forwardRef<HTMLDivElement, DropdownMenuTriggerProps>(
  ({ className, children, isOpen, setIsOpen, asChild, ...restProps }, ref) => {
    return (
      <div
        ref={ref}
        className={cn("inline-block", className)}
        onClick={() => setIsOpen?.(!isOpen)}
        {...restProps}
      >
        {children}
      </div>
    );
  }
)
DropdownMenuTrigger.displayName = "DropdownMenuTrigger"

interface DropdownMenuContentProps extends React.HTMLAttributes<HTMLDivElement> {
  align?: "start" | "center" | "end"
  isOpen?: boolean
  setIsOpen?: (open: boolean) => void
}

const DropdownMenuContent = React.forwardRef<HTMLDivElement, DropdownMenuContentProps>(
  ({ className, align = "center", isOpen, setIsOpen, children, ...restProps }, ref) => {
    const contentRef = React.useRef<HTMLDivElement>(null)

    // refをcontentRefと統合
    React.useImperativeHandle(ref, () => contentRef.current!, [])

    React.useEffect(() => {
      const handleClickOutside = (event: MouseEvent) => {
        if (contentRef.current && !contentRef.current.contains(event.target as Node)) {
          setIsOpen?.(false)
        }
      }

      if (isOpen) {
        document.addEventListener('mousedown', handleClickOutside)
        return () => document.removeEventListener('mousedown', handleClickOutside)
      }
    }, [isOpen, setIsOpen])

    if (!isOpen) return null

    return (
      <div
        ref={contentRef}
        className={cn(
          "absolute z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-md",
          align === "end" && "right-0",
          align === "start" && "left-0",
          align === "center" && "left-1/2 transform -translate-x-1/2",
          className
        )}
        {...restProps}
      >
        {children}
      </div>
    )
  }
)
DropdownMenuContent.displayName = "DropdownMenuContent"

interface DropdownMenuItemProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  inset?: boolean
}

const DropdownMenuItem = React.forwardRef<HTMLButtonElement, DropdownMenuItemProps>(
  ({ className, inset, ...props }, ref) => (
    <button
      ref={ref}
      className={cn(
        "relative flex w-full cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none transition-colors focus:bg-accent focus:text-accent-foreground data-[disabled]:pointer-events-none data-[disabled]:opacity-50",
        inset && "pl-8",
        className
      )}
      {...props}
    />
  )
)
DropdownMenuItem.displayName = "DropdownMenuItem"

export {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
}
